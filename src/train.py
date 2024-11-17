from typing import Optional, Tuple, Any
import time
import os
import torch
import tiktoken
from src.models.gpt2.model import GPT2
from src.models.gpt2.config import GPT2Config
from src.utils import set_seed, get_lr
from src.data.dataset import TextDataLoader
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from dotenv import load_dotenv
from knockknock import discord_sender
from src.data.hellaswag import evaluate as evaluate_hellaswag
from torch import Tensor
import torch.nn as nn
class TrainingConfig:
    def __init__(self):
        self.total_batch_size = 524288
        self.batch_size = 16
        self.sequence_length = 1024
        self.max_lr = 6e-4
        self.min_lr = self.max_lr * 0.1
        self.warmup_steps = 715
        self.total_steps = 19073

class GPTTrainer:
    def __init__(self, config: TrainingConfig, model: nn.Module, seed: int = 42, use_cuda_graphs: bool = False) -> None:
        self.config = config
        self.use_cuda_graphs = use_cuda_graphs
        self.setup_device(seed)
        self.setup_training_params()
        self.setup_model(model)
        self.setup_data()
        if self.master_process:
            self.setup_logging()
        if self.use_cuda_graphs and self.device_type == "cuda":
            self.setup_cuda_graphs()

    def setup_device(self, seed: int = 42) -> None:
        # DDP setup
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        if self.ddp:
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ.get('RANK'))
            self.ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
            self.ddp_world_size = int(os.environ.get('WORLD_SIZE'))
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.device = self._get_default_device()
            self.master_process = True

        self.device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        print(f"Using device: {self.device}")
        torch.set_float32_matmul_precision('high')
        set_seed(seed)

    def _get_default_device(self) -> str:
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def setup_training_params(self) -> None:
        self.grad_accumulation_steps = (
            self.config.total_batch_size // 
            (self.config.batch_size * self.config.sequence_length * self.ddp_world_size)
        )
        print(f"Grad accumulation steps: {self.grad_accumulation_steps}, effective batch size: {self.config.total_batch_size}")

    def setup_model(self, model: nn.Module) -> None:
        self.model = model
        self.model.train()
        self.model.to(self.device)
        self.model = torch.compile(self.model, mode="max-autotune", dynamic=False, fullgraph=True)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank], static_graph=True)
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model

        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.optimizer = self.raw_model.configure_optimizers(
            0.1, self.config.max_lr, self.device, self.master_process
        )

    def setup_cuda_graphs(self) -> None:
        # Warmup before capturing CUDA graph
        self.static_x = torch.zeros((self.config.batch_size, self.config.sequence_length), 
                                  dtype=torch.long, device=self.device)
        self.static_y = torch.zeros((self.config.batch_size, self.config.sequence_length), 
                                  dtype=torch.long, device=self.device)
        
        # Do a few warmup iterations
        for _ in range(3):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, loss = self.model(self.static_x, self.static_y)
                loss = loss / self.grad_accumulation_steps
            loss.backward()
        
        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                self.static_output, self.static_loss = self.model(self.static_x, self.static_y)
                self.static_loss = self.static_loss / self.grad_accumulation_steps
            self.static_loss.backward()

        # Additional validation graph setup
        self.val_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.val_graph):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, self.static_val_loss = self.model(self.static_x, self.static_y)

    def setup_data(self) -> None:
        self.data_loader = TextDataLoader(
            self.config.batch_size, 
            self.config.sequence_length,
            "edu_fineweb10B",
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size
        )
        self.val_loader = TextDataLoader(
            self.config.batch_size,
            self.config.sequence_length,
            single_file='edufineweb_val.safetensors',
            process_rank=self.ddp_rank,
            num_processes=self.ddp_world_size
        )

    def setup_logging(self) -> None:
        load_dotenv()
        wandb.login()
        os.makedirs('logs', exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.log_file = f'logs/training_log_{timestamp}.txt'
        
        with open(self.log_file, 'w') as f:
            f.write('step,train_loss,val_loss,learning_rate,grad_norm,time\n')
        
        wandb.init(
            project="gpt2-training",
            config=self.config.__dict__
        )

    def train_step(self, step: int) -> Tuple[Tensor, float, Tensor, float]:
        start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        loss_accum = 0

        for mini_step in range(self.grad_accumulation_steps):
            loss_accum += self._process_batch(mini_step)

        if self.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Optimize
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        lr = get_lr(self.config.warmup_steps, self.config.total_steps, 
                   self.config.max_lr, self.config.min_lr, step)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

        return loss_accum, lr, norm, time.time() - start

    def _process_batch(self, mini_step: int) -> Tensor:
        x, y = self.data_loader.get_item()
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        
        if self.ddp:
            self.model.require_backward_grad_sync = (
                mini_step == self.grad_accumulation_steps - 1
            )
        
        if self.use_cuda_graphs and self.device_type == "cuda":
            # Update static buffers and replay graph
            self.static_x.copy_(x)
            self.static_y.copy_(y)
            self.graph.replay()
            return self.static_loss.detach()
        else:
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs, loss = self.model(x, y)
            
            loss = loss / self.grad_accumulation_steps
            loss.backward()
            return loss.detach()

    def validate(self) -> Tensor:
        self.model.eval()
        val_loss_accum = 0.0
        val_loss_steps = 20

        self.val_loader.reset()

        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = self.val_loader.get_single_file_item()
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                if self.use_cuda_graphs and self.device_type == "cuda":
                    self.static_x.copy_(x)
                    self.static_y.copy_(y)
                    self.val_graph.replay()
                    val_loss_accum += self.static_val_loss.detach() / val_loss_steps
                else:
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        _, loss = self.model(x, y)
                    val_loss_accum += loss.detach() / val_loss_steps

        if self.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        self.model.train()
        return val_loss_accum

    def train(self) -> None:
        try:
            for step in range(self.config.total_steps):
                loss_accum, lr, norm, dt = self.train_step(step)
                
                if step % 250 == 0:
                    self._handle_validation_step(step, loss_accum, lr, norm, dt)

            if self.master_process:
                wandb.finish()
                self._send_discord_notification('Training finished')

        finally:
            if self.ddp:
                destroy_process_group()

    def _handle_validation_step(self, step: int, loss_accum: Tensor, lr: float, 
                              norm: Tensor, dt: float) -> None:
        val_loss = self.validate()
        last_step = (step == self.config.total_steps - 1)
        
        if self.master_process:
            self._log_metrics(step, loss_accum, lr, norm, dt, val_loss)
            
            if (step % 1000 == 0 or last_step) and step != 0:
                hellaswag_acc = evaluate_hellaswag(self.raw_model, self.device)
                self._log_hellaswag(step, hellaswag_acc)

    @discord_sender(webhook_url=os.environ.get('DISCORD_WEBHOOK_URL'))
    def _send_discord_notification(self, message: str) -> str:
        return message

    def _log_metrics(self, step: int, loss_accum: Tensor, lr: float, 
                    norm: Tensor, dt: float, val_loss: Tensor) -> None:
        tok_sec = (self.config.batch_size * self.config.sequence_length * 
                  self.grad_accumulation_steps * self.ddp_world_size) / dt

        print(f'Step {step} | loss: {loss_accum.item():.6f} | val_loss: {val_loss.item():.4f} | '
              f'lr: {lr:.5f} | norm: {norm:.4f} | time: {dt*1000:.2f}ms | '
              f'tokens/sec: {tok_sec:.2f}')

        with open(self.log_file, "a") as f:
            f.write(f"{step},{loss_accum.item():.4f},{val_loss.item():.4f},"
                   f"{lr:.5f},{norm:.4f},{dt*1000:.2f}\n")

        wandb.log({
            "train/loss": loss_accum.item(),
            "val/loss": val_loss.item(),
            "train/learning_rate": lr,
            "train/grad_norm": norm,
            "performance/step_time_ms": dt * 1000,
            "performance/tokens_per_second": tok_sec,
        }, step=step)

    def _log_hellaswag(self, step: int, accuracy: float) -> None:
        print(f"hellaswag accuracy: {accuracy:.4f}")
        with open(self.log_file, "a") as f:
            f.write(f"{step} hellaswag {accuracy:.4f}\n")
        wandb.log({"eval/hellaswag_accuracy": accuracy}, step=step)
# Configuration class

# Usage
if __name__ == "__main__":
    config = TrainingConfig()
    model = GPT2(GPT2Config(vocab_size=50304))
    trainer = GPTTrainer(config, model)
    trainer.train()
