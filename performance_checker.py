import time
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from typing import Optional, Tuple, Any
import torch.cuda.amp as amp

class PerformanceTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        total_batch_size: int,
        micro_batch_size: int,
        sequence_length: int,
        enable_cuda_graph: bool = True,
        use_fused_adam: bool = True,
        use_apex_ddp: bool = False,  # Requires apex installation
        profile_cuda: bool = False,
    ):
        """
        Initialize performance-optimized trainer.
        
        Args:
            model: Your model (will be wrapped in DDP if distributed)
            optimizer: Your optimizer (will be wrapped in AMP if using mixed precision)
            total_batch_size: Global batch size across all GPUs
            micro_batch_size: Batch size per GPU per step
            sequence_length: Length of input sequences
            enable_cuda_graph: Whether to use CUDA graphs for static parts
            use_fused_adam: Whether to use CUDA Fused Adam
            use_apex_ddp: Whether to use NVIDIA Apex DDP instead of PyTorch DDP
            profile_cuda: Whether to profile CUDA operations
        """
        self.setup_distributed()
        self.setup_device()
        
        # Basic configuration
        self.total_batch_size = total_batch_size
        self.micro_batch_size = micro_batch_size
        self.sequence_length = sequence_length
        self.enable_cuda_graph = enable_cuda_graph and self.device_type == 'cuda'
        self.profile_cuda = profile_cuda
        
        # Calculate gradient accumulation steps
        self.grad_accumulation_steps = total_batch_size // (micro_batch_size * self.world_size)
        assert total_batch_size % (micro_batch_size * self.world_size) == 0, \
            'Total batch size must be divisible by (micro_batch_size * world_size)'
            
        # Model setup with optimizations
        self.setup_model(model, use_apex_ddp)
        
        # Optimizer setup with optimizations
        self.setup_optimizer(optimizer, use_fused_adam)
        
        # AMP (Automatic Mixed Precision) setup
        self.setup_amp()
        
        # CUDA Graphs setup
        self.setup_cuda_graphs() if self.enable_cuda_graph else None
        
        if self.master_process:
            print(f"Effective batch size: {total_batch_size}")
            print(f"Gradient accumulation steps: {self.grad_accumulation_steps}")
            print(f"Device: {self.device} ({self.device_type})")
            if self.ddp:
                print(f"World size: {self.world_size}")

    def setup_distributed(self):
        """Setup distributed training environment"""
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend='nccl')
            self.rank = int(os.environ.get('RANK'))
            self.local_rank = int(os.environ.get('LOCAL_RANK'))
            self.world_size = int(os.environ.get('WORLD_SIZE'))
            self.device = f'cuda:{self.local_rank}'
            self.master_process = self.rank == 0
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.master_process = True

    def setup_device(self):
        """Setup device and optimization flags"""
        if self.device.startswith('cuda'):
            self.device_type = 'cuda'
            torch.cuda.set_device(self.device)
            # Enable tensor cores
            torch.set_float32_matmul_precision('high')
            # Optimize memory allocator
            torch.cuda.empty_cache()
        else:
            self.device_type = 'cpu'

    def setup_model(self, model: torch.nn.Module, use_apex_ddp: bool):
        """Setup model with optimizations"""
        model = model.to(self.device)
        
        # Apply torch.compile with optimal settings
        model = torch.compile(
            model,
            mode='max-autotune',  # Use the most aggressive optimization
            fullgraph=True,  # Enable full graph optimization
            dynamic=False,  # Disable dynamic shapes for better optimization
        )
        
        if self.ddp:
            if use_apex_ddp:
                try:
                    from apex.parallel import DistributedDataParallel as ApexDDP
                    model = ApexDDP(model)
                except ImportError:
                    print("Apex not found, falling back to PyTorch DDP")
                    model = DDP(
                        model,
                        device_ids=[self.local_rank],
                        static_graph=True  # Enable static graph optimization
                    )
            else:
                model = DDP(
                    model,
                    device_ids=[self.local_rank],
                    static_graph=True
                )
        
        self.model = model

    def setup_optimizer(self, optimizer: torch.optim.Optimizer, use_fused_adam: bool):
        """Setup optimizer with optimizations"""
        # If using Adam, optionally replace with CUDA Fused Adam
        if use_fused_adam and isinstance(optimizer, torch.optim.Adam):
            try:
                from apex.optimizers import FusedAdam
                optimizer_params = optimizer.defaults
                optimizer = FusedAdam(self.model.parameters(), **optimizer_params)
            except ImportError:
                print("Apex not found, using standard Adam")
        
        self.optimizer = optimizer

    def setup_amp(self):
        """Setup Automatic Mixed Precision"""
        self.autocast = nullcontext if self.device_type == 'cpu' else \
                       lambda: torch.autocast(device_type=self.device_type, dtype=torch.bfloat16)

    def setup_cuda_graphs(self):
        """Setup CUDA graphs for static parts of the computation"""
        if not self.enable_cuda_graph:
            return
            
        # Warmup before capturing CUDA graph
        self.model.train()
        x = torch.randint(0, 100, (self.micro_batch_size, self.sequence_length), device=self.device)
        y = torch.randint(0, 100, (self.micro_batch_size, self.sequence_length), device=self.device)
        
        # Capture forward pass
        self.static_x = torch.zeros_like(x)
        self.static_y = torch.zeros_like(y)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            for _ in range(3):  # Warmup
                with self.autocast():
                    outputs = self.model(self.static_x)
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), self.static_y.view(-1))
                    loss = loss / self.grad_accumulation_steps
                loss.backward()
        
        torch.cuda.current_stream().wait_stream(s)
        
        # Create CUDA graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with self.autocast():
                self.static_outputs = self.model(self.static_x)
                self.static_loss = torch.nn.functional.cross_entropy(
                    self.static_outputs.view(-1, self.static_outputs.size(-1)), 
                    self.static_y.view(-1)
                )
                self.static_loss = self.static_loss / self.grad_accumulation_steps
            self.static_loss.backward()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """
        Perform single training step with optimizations.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            tuple: (loss value, tokens/second)
        """
        start_time = time.time()
        
        # Optional CUDA profiling
        profile_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) if self.profile_cuda else nullcontext()
        
        with profile_context:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)  # Slightly faster than set_to_zero=False
            loss_accum = 0
            
            for micro_step in range(self.grad_accumulation_steps):
                # Get micro batch
                micro_x = x[micro_step * self.micro_batch_size:(micro_step + 1) * self.micro_batch_size]
                micro_y = y[micro_step * self.micro_batch_size:(micro_step + 1) * self.micro_batch_size]
                
                # Move to device with non_blocking for potential speedup
                micro_x = micro_x.to(self.device, non_blocking=True)
                micro_y = micro_y.to(self.device, non_blocking=True)
                
                if self.enable_cuda_graph:
                    # Update static buffers and replay graph
                    self.static_x.copy_(micro_x)
                    self.static_y.copy_(micro_y)
                    self.graph.replay()
                    loss = self.static_loss
                else:
                    # Regular forward pass with automatic mixed precision
                    with self.autocast():
                        outputs = self.model(micro_x)
                        loss = torch.nn.functional.cross_entropy(
                            outputs.view(-1, outputs.size(-1)), 
                            micro_y.view(-1)
                        )
                        loss = loss / self.grad_accumulation_steps
                    
                    # Backward pass
                    loss.backward()
                
                loss_accum += loss.detach()
            
            # Synchronize loss across GPUs if using DDP
            if self.ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            
            # Optimizer step with gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        # Calculate throughput
        torch.cuda.synchronize()
        end_time = time.time()
        tokens_per_sec = (self.total_batch_size * self.sequence_length) / (end_time - start_time)
        
        return loss_accum.item(), tokens_per_sec

    def cleanup(self):
        """Cleanup distributed training resources"""
        if self.ddp:
            destroy_process_group()
            
    def get_memory_stats(self):
        """Get current GPU memory statistics"""
        if self.device_type != 'cuda':
            return {}
            
        return {
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved(self.device) / 1024**2,  # MB
            'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**2,  # MB
        }
# Example usage:
"""
model = YourModel()
optimizer = torch.optim.AdamW(model.parameters())
trainer = PerformanceTrainer(
    model=model,
    optimizer=optimizer,
    total_batch_size=524288,
    micro_batch_size=16,
    sequence_length=1024,
    enable_cuda_graph=True,
    use_fused_adam=True
)

try:
    for step in range(num_steps):
        x, y = get_batch()  # Your data loading function
        loss, tokens_per_sec = trainer.train_step(x, y)
        
        if trainer.master_process and step % 10 == 0:
            memory_stats = trainer.get_memory_stats()
            print(f"Step {step}: loss={loss:.4f}, throughput={tokens_per_sec:.2f} tokens/sec")
            print(f"Memory: {memory_stats}")
finally:
    trainer.cleanup()
"""