import os
import argparse
import time
import warnings
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

from transformers import AutoTokenizer, get_scheduler

# 假设你的模型和数据集代码分别在 k_model.py 和 dataset.py 文件中
from k_model import ModelConfig, Transformer
from dataset import PretrainDatasetDynamic as PretrainDataset

import swanlab

warnings.filterwarnings('ignore')


def Logger(content):
    """使用 tqdm.write 打印日志，避免破坏进度条"""
    tqdm.write(content)

def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    """保存完整的训练状态以便恢复（单卡版）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(), # 直接保存模型状态
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)
    Logger(f"✅ Checkpoint saved to {path} (Step: {step}, Loss: {loss:.4f})")


def train_model():
    """主训练函数"""
    
    # --- 初始化组件 ---
    Logger("Initializing model and tokenizer...")
    model, tokenizer = init_model()
    
    Logger("Initializing dataset and dataloader...")
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, pin_memory=True,
        drop_last=True, shuffle=True, num_workers=args.num_workers
    )

    Logger("Initializing optimizer, scheduler, and scaler...")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 仅在dtype为float16时启用GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    num_training_steps_per_epoch = len(train_loader) // args.accumulation_steps
    total_training_steps = args.epochs * num_training_steps_per_epoch
    
    scheduler = get_scheduler(
        name="cosine", optimizer=optimizer,
        num_warmup_steps=args.warmup_iters,
        num_training_steps=total_training_steps
    )

    # --- 从 Checkpoint 恢复 ---
    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        try:
            Logger(f"Resuming from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=args.device)
            
            # 直接加载状态字典
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_step = checkpoint.get('step', 0) + 1
            Logger(f"✅ Successfully resumed from step {start_step}. Last loss: {checkpoint.get('loss', 'N/A'):.4f}")
        except Exception as e:
            Logger(f"⚠️ Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        Logger("No checkpoint found. Starting training from scratch.")

    # --- 训练循环 ---
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    current_step = start_step
    
    for epoch in range(args.epochs):
        # 计算当前epoch的起始step在dataloader中的位置
        start_iter = start_step % len(train_loader) if epoch == (start_step // len(train_loader)) else 0
        
        progress_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader), 
            desc=f"Epoch {epoch+1}/{args.epochs}",
            initial=start_iter
        )

        # 如果需要，跳过已完成的批次
        if start_iter > 0:
            # Dataloader的迭代器不支持直接seek，所以我们重新创建一个迭代器并快进
            train_iterator = iter(train_loader)
            for _ in range(start_iter):
                next(train_iterator)
            progress_bar = tqdm(
                enumerate(train_iterator, start=start_iter),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{args.epochs}",
                initial=start_iter
            )
        
        for step_in_epoch, batch in progress_bar:
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)

            with ctx:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (current_step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=f"{loss.item() * args.accumulation_steps:.4f}", lr=f"{current_lr:.7f}")

            if (current_step + 1) % args.log_interval == 0:
                if args.use_swanlab:
                    swanlab.log({"loss": loss.item() * args.accumulation_steps, "lr": current_lr}, step=current_step)

            if (current_step + 1) % args.save_interval == 0:
                save_checkpoint(
                    model, optimizer, scheduler, current_step,
                    loss.item() * args.accumulation_steps,
                    path=os.path.join(args.save_dir, f"checkpoint_step_{current_step+1}.pt")
                )
            
            current_step += 1

def init_model():
    """初始化模型和分词器（单卡版）"""
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')
    model = Transformer(lm_config)
    
    # 直接将模型移动到指定设备
    model = model.to(args.device)
    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining (Single-Card)")
    
    # 目录和路径参数
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="最终模型输出目录")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="训练检查点保存目录")
    parser.add_argument("--data_path", type=str, default="./seq_monkey_datawhale.jsonl", help="训练数据路径")
    parser.add_argument("--resume_from", type=str, default=None, help="从指定的checkpoint文件恢复训练")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="最大学习率")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW的权重衰减")
    
    # 设备和精度
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备 (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16", help="数据类型 ('float16', 'bfloat16', 'float32')")
    
    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=2000, help="学习率预热迭代次数")
    
    # 日志和保存
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # 数据加载
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载的工作进程数")

    # 实验跟踪
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")

    args = parser.parse_args()

    # 环境和配置设置
    if args.use_swanlab:
        swanlab.init(project="Happy-LLM", experiment_name="Pretrain-215M-SingleCard", config=args)

    lm_config = ModelConfig(dim=1024, n_layers=18)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.manual_seed(42)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=pt_dtype)

    # 启动训练
    train_model()