import time
import os
import torch
import tiktoken
from layers import GPT2
from configs import GPTConfig
from utils import set_seed, get_lr
from dataset import TextDataLoader
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import safetensors.torch as safetorch
from hellaswag import evaluate as evaluate_hellaswag
import wandb
from dotenv import load_dotenv


# 1. Device and DDP setup
ddp = int(os.environ.get('DDP', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get('RANK'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = 'cpu'
    master_process = True
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'

device_type = "cuda" if device.startswith("cuda") else "cpu"
torch.set_float32_matmul_precision('high')
print(f'Using device: {device}')
set_seed(42)

# 2. Training configuration
total_batch_size = 524288
batch_size = 16
sequence_length = 1024
assert total_batch_size % (batch_size * sequence_length) == 0, 'Total batch size must be divisible by batch size * sequence length'
grad_accumulation_steps = total_batch_size // (batch_size * sequence_length * ddp_world_size)
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
total_steps = 19073

if master_process:
    print(f'Effective batch size: {total_batch_size}, grad accumulation steps: {grad_accumulation_steps}')

# 3. Model, tokenizer, and optimizer setup
model = GPT2(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
else:
    raw_model = model

tokenizer = tiktoken.get_encoding('gpt2')
optimizer = raw_model.configure_optimizers(0.1, max_lr, device)

#todo
data_loader = TextDataLoader(batch_size, sequence_length, "edu_fineweb10B", process_rank=ddp_rank, num_processes=ddp_world_size)
val_loader = TextDataLoader(batch_size, sequence_length, single_file='edufineweb_val.safetensors', process_rank=ddp_rank, num_processes=ddp_world_size)

# 5. Logging setup
log_file = None
log_dir = None
if master_process:
    load_dotenv()
    wandb.login()
    os.makedirs('logs', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')

    log_file = f'logs/training_log_{timestamp}.txt'
    with open(log_file, 'w') as f:
        f.write('step,train_loss,val_loss,learning_rate,grad_norm,time\n')
    # Initialize wandb
    wandb.init(
        project="gpt2-training",
        config={
            "total_batch_size": total_batch_size,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        }
    )

for step in range(total_steps):
    last_step = (step == total_steps - 1)
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0
    for mini_step in range(grad_accumulation_steps):
        x, y = data_loader.get_item()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if ddp:
            model.require_backward_grad_sync = (mini_step == grad_accumulation_steps - 1)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs, loss = model(x, y)
        loss = loss / grad_accumulation_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(warmup_steps, total_steps, max_lr, min_lr, step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    if step % 250 == 0:
        model.eval()
        val_loader.reset()  # Reset the validation loader
        val_loss_accum = 0.0
        val_loss_steps = 20
        with torch.no_grad():
            for _ in range(val_loss_steps):
                x, y = val_loader.get_single_file_item()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            # Add wandb logging for validation
            wandb.log({
                "val/loss": val_loss_accum.item(),
            }, step=step)

        # Add HellaSwag evaluation
        if master_process and (step % 1000 == 0 or last_step) and step != 0:
            hellaswag_acc = evaluate_hellaswag(raw_model, device)
            print(f"hellaswag accuracy: {hellaswag_acc:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hellaswag {hellaswag_acc:.4f}\n")
            # Add wandb logging for HellaSwag
            wandb.log({
                "eval/hellaswag_accuracy": hellaswag_acc,
            }, step=step)

        model.train()
    torch.cuda.synchronize()
    end = time.time()
    dt = (end - start) * 1000 # in ms
    tok_sec = (batch_size * sequence_length * grad_accumulation_steps * ddp_world_size) / (end - start)
    if master_process:
        print(f'Step {step} | loss: {loss_accum.item():.6f}| lr: {lr:.4f} | norm: {norm:.4f} | time: {dt:.2f}ms | tokens/sec: {tok_sec:.2f}')
        # Add wandb logging
        wandb.log({
            "train/loss": loss_accum.item(),
            "train/learning_rate": lr,
            "train/grad_norm": norm,
            "performance/step_time_ms": dt,
            "performance/tokens_per_second": tok_sec,
        }, step=step)

if master_process:
    wandb.finish()

if ddp:
    destroy_process_group()