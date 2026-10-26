"""Khung huấn luyện PyTorch: DDP trên 2×T4, AMP fp16, checkpoint/resume, tôn trọng deadline.

Agent thay phần build_data() và build_model() (đánh dấu bên dưới) cho bài toán thật.
Chạy được cả khi:
  - torchrun --nproc_per_node=2 train.py ...   (DDP, 2 GPU)
  - python train.py ...                        (1 GPU hoặc CPU)
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch-size", type=int, default=64, help="batch MỖI GPU")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--weight-decay", type=float, default=0.01)
ap.add_argument("--grad-accum", type=int, default=1)
ap.add_argument("--max-steps", type=int, default=0, help=">0: dừng sớm (smoke test)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--num-workers", type=int, default=2)
args = ap.parse_args()

SMOKE = os.environ.get("LABRAT_SMOKE") == "1"
OUT = Path(os.environ.get("LABRAT_OUTPUT_DIR", "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
DEADLINE = float(os.environ.get("LABRAT_DEADLINE", "inf"))
CKPT = OUT / "checkpoint.pt"

# ---------------- khởi tạo phân tán ----------------
DDP = int(os.environ.get("WORLD_SIZE", "1")) > 1
if DDP:
    dist.init_process_group("nccl")
    RANK, WORLD = dist.get_rank(), dist.get_world_size()
    LOCAL = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(LOCAL)
    DEVICE = torch.device("cuda", LOCAL)
else:
    RANK, WORLD = 0, 1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAIN = RANK == 0
USE_AMP = DEVICE.type == "cuda"          # fp16 trên T4
torch.manual_seed(args.seed + RANK)


def log(*a):
    if MAIN:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# =============== DỮ LIỆU & MÔ HÌNH — agent thay phần này cho bài thật ===============
class ToyDataset(Dataset):
    """Dữ liệu giả để kiểm tra pipeline. Thay bằng dataset thật (đọc từ /kaggle/input)."""

    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 32, generator=g)
        self.y = (self.x[:, :4].sum(1) > 0).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def build_data():
    n = 512 if SMOKE else 50_000
    return ToyDataset(n, 0), ToyDataset(n // 5, 1)


def build_model():
    return nn.Sequential(nn.Linear(32, 256), nn.GELU(), nn.Linear(256, 2))


def evaluate_metric(logits, labels) -> float:
    """Metric chính của cuộc thi. Ví dụ: accuracy. Phải khớp metric_direction trong job.yaml."""
    return (logits.argmax(1) == labels).float().mean().item()
# ===================================================================================


train_ds, valid_ds = build_data()
train_sampler = DistributedSampler(train_ds, shuffle=True) if DDP else None
train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                      shuffle=train_sampler is None, num_workers=args.num_workers,
                      pin_memory=DEVICE.type == "cuda", drop_last=True, persistent_workers=args.num_workers > 0)
valid_dl = DataLoader(valid_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers)

model = build_model().to(DEVICE)
if DDP:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL])
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
steps_per_epoch = math.ceil(len(train_dl) / args.grad_accum)
total_steps = max(1, steps_per_epoch * args.epochs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
loss_fn = nn.CrossEntropyLoss()

start_epoch, best, history = 0, None, []
if CKPT.exists():  # resume (ví dụ khi checkpoint được đưa vào qua dataset)
    ck = torch.load(CKPT, map_location=DEVICE)
    (model.module if DDP else model).load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
    log(f"resume từ epoch {start_epoch}")


def save_ckpt(epoch):
    if MAIN:
        torch.save({"model": (model.module if DDP else model).state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "best": best, "history": history}, CKPT)


@torch.no_grad()
def validate():
    model.eval()
    logits, labels = [], []
    for x, y in valid_dl:
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            logits.append(model(x.to(DEVICE, non_blocking=True)).float().cpu())
        labels.append(y)
    model.train()
    return evaluate_metric(torch.cat(logits), torch.cat(labels))


t0, step, stopped = time.time(), 0, None
for epoch in range(start_epoch, args.epochs):
    if train_sampler:
        train_sampler.set_epoch(epoch)
    for i, (x, y) in enumerate(train_dl):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            loss = loss_fn(model(x), y) / args.grad_accum
        scaler.scale(loss).backward()
        if (i + 1) % args.grad_accum == 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if sched.last_epoch < total_steps - 1:
                sched.step()
            step += 1
        if args.max_steps and step >= args.max_steps:
            stopped = "max_steps"
            break
        if i % 20 == 0:  # đồng bộ cờ deadline giữa các rank để không rank nào bị treo chờ
            flag = torch.tensor([1.0 if time.time() > DEADLINE else 0.0], device=DEVICE)
            if DDP:
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item() > 0:
                stopped = "deadline"
                break

    # mọi rank cùng đánh giá trên toàn bộ valid (đơn giản, chấp nhận trùng lặp)
    score = validate()
    history.append({"epoch": epoch, "score": score, "loss": float(loss.item() * args.grad_accum),
                    "elapsed_min": round((time.time() - t0) / 60, 1)})
    if best is None or score > best:   # đảo dấu nếu metric càng thấp càng tốt
        best = score
        if MAIN:
            torch.save((model.module if DDP else model).state_dict(), OUT / "best_model.pt")
    save_ckpt(epoch)
    log(f"epoch {epoch} score={score:.5f} best={best:.5f}")
    if stopped:
        log(f"dừng sớm: {stopped}")
        break

if MAIN:
    Path(os.environ.get("LABRAT_METRICS_PATH", OUT / "metrics_local.json")).write_text(json.dumps({
        "cv": best, "history": history, "world_size": WORLD, "stopped": stopped,
        "minutes": round((time.time() - t0) / 60, 1), "amp_fp16": USE_AMP,
    }, indent=2))
if DDP:
    dist.barrier()
    dist.destroy_process_group()
