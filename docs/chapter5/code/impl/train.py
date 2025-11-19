import csv
import os
import argparse
import warnings
import math
import torch
from torch import optim
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from transformers import AutoTokenizer, get_scheduler

from NanoLlama import ModelConfig, NanoLlama
from dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    """使用 tqdm.write 打印日志，避免破坏进度条"""
    tqdm.write(str(content))

def create_log_file(log_path):
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "learning_rate"])

def save_checkpoint(model, optimizer, scheduler, scaler, step, loss, path):
    """保存完整的训练状态以便恢复（单卡版）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)
    Logger(f"✅ Checkpoint saved to {path} (Step: {step}, Loss: {loss:.4f})")


def init_model(args, lm_config):
    """显式传入 args 与 lm_config，返回 model 与 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
    lm_config.vocab_size = tokenizer.vocab_size
    model = NanoLlama(lm_config)
    model = model.to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


def train_model(args, lm_config, ctx):
    """主训练函数（显式接收 args, lm_config, ctx）"""

    # --- 初始化组件 ---
    Logger("Initializing model and tokenizer...")
    model, tokenizer = init_model(args, lm_config)

    Logger("Initializing dataset and dataloader...")
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size, 
        pin_memory=True,
        drop_last=True, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=train_ds.collate_fn,
        persistent_workers=True
    )

    Logger("Creating log file...")
    create_log_file(args.log_path)

    Logger("Initializing optimizer, scheduler, and scaler...")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-9)
    # 只有在 float16 且 使用 CUDA 的情况下启用 GradScaler（bfloat16 不需要 scaler）
    use_scaler = (args.dtype == 'float16') and ("cuda" in args.device)
    scaler = GradScaler(enabled=use_scaler, device=args.device)

    # 更稳健地计算 total training steps（考虑梯度累积）
    total_training_steps = math.ceil(len(train_loader) * args.epochs / args.accumulation_steps)

    scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=args.warmup_iters,
        num_training_steps=total_training_steps
    )

    loss_fct = torch.nn.CrossEntropyLoss()

    # --- 从 Checkpoint 恢复 ---
    start_step = 1
    if args.resume_from and os.path.exists(args.resume_from):
        try:
            Logger(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=args.device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            start_step = int(checkpoint.get('step', 0)) + 1
            Logger(f"✅ Successfully resumed from step {start_step}. Last loss: {checkpoint.get('loss', 'N/A'):.4f}")
        except Exception as e:
            Logger(f"⚠️ Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        Logger("No checkpoint found. Starting training from scratch.")

    # --- 训练循环 ---
    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = start_step
    accumulation_counter = 0

    for epoch in range(args.epochs):
        # 每个 epoch 都重新 shuffle（DataLoader shuffle=True），不恢复 dataloader 内部位置
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for step_in_epoch, batch in progress_bar:
            # 如果当前 step 已经超过或达到总步数，结束训练
            if global_step > total_training_steps:
                Logger("Reached total training steps. Exiting training loop.")
                return

            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)

            # 使用提供的 autocast 上下文（ctx），例如 bfloat16/float16_autocast
            with ctx:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                # 语言模型下 shift
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            # 梯度累积步到达后再更新
            if accumulation_counter % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                accumulation_counter = 0

                current_lr = scheduler.get_last_lr()[0]
                loss_value = loss.item() * args.accumulation_steps

                progress_bar.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{current_lr:.7f}",
                    step=f"{global_step}/{total_training_steps}"
                )

                # 日志记录
                if global_step % args.log_interval == 0:
                    with open(args.log_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([global_step, loss_value, current_lr])

                # 保存 checkpoint（按绝对 step）
                if global_step % args.save_interval == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step,
                        loss_value,
                        path=os.path.join(args.save_dir, f"step_{global_step}.pt")
                    )


    Logger("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoLlama Pretrain (Single-Card)")

    # 目录和路径参数
    parser.add_argument("--out_dir", type=str, default="ouput", help="结果输出目录")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="训练检查点保存目录")
    parser.add_argument("--data_path", type=str, default="data/pretrain_sample_data.jsonl", help="训练数据路径")
    parser.add_argument("--log_path", type=str, default="ouput/training_log.csv", help="训练日志保存路径")
    parser.add_argument("--resume_from", type=str, default=None, help="从指定的checkpoint文件恢复训练")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="最大学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW的权重衰减")

    # 设备和精度
    parser.add_argument("--device", type=str, default=None, help="训练设备 (e.g., 'cuda', 'cpu')")
    parser.add_argument("--dtype", type=str, default=None, help="数据类型 ('float16', 'bfloat16', 'float32')")

    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=4000, help="学习率预热迭代次数")

    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 数据加载
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")

    args = parser.parse_args()

    # 如果用户没传 device，就基于当前环境决定
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # 安全判断 bfloat16 支持（放在解析之后并用 try/except）
    if args.dtype is None:
        pt_supports_bf16 = False
        if "cuda" in args.device:
            print("Using device:", torch.cuda.current_device(), torch.cuda.get_device_name())
            try:
                pt_supports_bf16 = torch.cuda.is_bf16_supported()
            except Exception as e:
                Logger(f"Warning: torch.cuda.is_bf16_supported() failed: {e}. Falling back to float16.")
                pt_supports_bf16 = False
        args.dtype = "bfloat16" if pt_supports_bf16 else "float16"

    print(args)

    # exit(0)

    lm_config = ModelConfig(dim=1024, n_layers=18)
    # lm_config = ModelConfig(dim=512, n_layers=6)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    torch.manual_seed(42)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = autocast(device_type=device_type, dtype=pt_dtype)

    # 启动训练（显式传参）
    train_model(args, lm_config, ctx)