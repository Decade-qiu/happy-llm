import csv
from numpy import pad
import torch, os
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import math
import random
from typing import List, Tuple
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from model3 import ModelArgs, Transformer, get_padding_mask, get_causal_mask

# =======================
# 🧩 1. 设备设置
# =======================
if torch.cuda.is_available():
    device = "cuda"
    print("Using: CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    device = "mps"
    print("Using: MPS (Apple Silicon GPU)")
else:
    device = "cpu"
    print("Using: CPU")


# =======================
# 🧩 3. Tokenizer 设置
# =======================

def build_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    special_tokens = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>"
    }
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def save_checkpoint(model, optimizer, scheduler, step, loss, path="checkpoint.pt", note=None):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "note": note or "",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)
    print(f"✅ 模型与训练进度已保存到 {path} (step={step}, loss={loss:.4f})")


# =======================
# 🧩 5. 学习率调度器
# =======================

def get_lr_scheduler(optimizer, warmup_steps=4000, d_model=512, initial_lr=1e-7):
    factor = 1.0 / math.sqrt(d_model)
    def lr_lambda(step):
        if step == 0:
            return initial_lr
        step_f = float(step)
        return factor * min(step_f * (warmup_steps ** (-1.5)), step_f ** (-0.5))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =======================
# 🧩 7. 训练函数（封装）
# =======================

def train_model(
    data_path="/home/qzj/A/RAG/Transformer/impl/data/wmt_zh_en_training_corpus_cleaned.csv",
    batch_size=64,
    accum_steps=4,
    max_len=512,
    lr=1.0,
    warmup_steps=4000,
    ckpt_path=None,
    log_path="/home/qzj/A/RAG/Transformer/impl/data/training_log.csv",
    log_step=10000,
    max_steps=100000,
    max_epoch=3
):
    tokenizer = build_tokenizer()
    pad_idx = tokenizer.pad_token_id
    print(f"Tokenizer Init Finished ({len(tokenizer)})")

    args = ModelArgs()
    args.src_vocab_size = len(tokenizer)
    args.tgt_vocab_size = len(tokenizer)
    args.src_tgt_emb_shared = True

    train_dataset = ToyTranslationDataset(
        data_path, tokenizer, max_len=max_len,
        mode="train", val_ratio=0.01
    )
    val_dataset = ToyTranslationDataset(
        data_path, tokenizer, max_len=max_len,
        mode="val", val_ratio=0.01
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_idx, pad_idx)
    )
    print(f"train_loader Init Finished ({len(train_loader)})")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_idx, pad_idx)
    )
    print(f"val_loader Init Finished ({len(val_loader)})")

    model = Transformer(args).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, d_model=args.dim)

    cnt = 1
    best_loss = float("inf")

    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "learning_rate"])

    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            cnt = ckpt.get('step', 0) + 1
            best_loss = ckpt.get('loss', best_loss)
            print(f"成功加载 checkpoint '{ckpt_path}'，从第 {cnt} 轮继续训练。")
        except Exception as e:
            print(f"⚠️ 无法加载 checkpoint ({ckpt_path})，错误：{e}")
    else:
        print("未找到有效的 checkpoint，从头开始训练。")
        cnt = 1
        best_loss = float("inf")

    log_loss, avg_loss = 0, 0

    print("开始训练...")
    model.train()
    optimizer.zero_grad()

    for epoch in range(0, max_epoch):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epoch}", ncols=100)
        
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = get_padding_mask(src, pad_idx).to(device)
            tgt_padding_mask = get_padding_mask(tgt_input, pad_idx).to(device)
            tgt_causal_mask = get_causal_mask(tgt_input.size(1)).to(device)
            tgt_mask = tgt_padding_mask & tgt_causal_mask
            
            # Mixed bfloat16
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            loss = loss / accum_steps
            loss.backward()
            log_loss += loss.item() * accum_steps

            pbar.set_postfix({
                'loss': f'{loss.item() * accum_steps:.4f}', 
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

            if cnt % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if cnt % 100 == 0:
                avg_loss = log_loss / 100
                with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([cnt, avg_loss, 0, scheduler.get_last_lr()[0]])
                log_loss = 0

            if cnt % log_step == 0:
                val_loss = evaluate(model, val_loader, criterion, pad_idx)
                lr_value = scheduler.get_last_lr()[0]
                print(f"Epoch {cnt}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, lr={lr_value:.6f}")

                with open(log_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([cnt, avg_loss, val_loss, lr_value])

                if 1:
                    path = f"checkpoints/transformer_epoch_{cnt}.pt"
                    print(f"Save the checkpoint in {path}")
                    save_checkpoint(
                        model, optimizer, scheduler,
                        cnt, val_loss,
                        path
                    )

                if val_loss < best_loss:
                    print(f"Save best model with loss {val_loss:.4f}")
                    best_loss = val_loss
                    torch.save(model.state_dict(), "translation_model.pt")

            if cnt >= max_steps:
                print("训练完成 ✅")
                return model, tokenizer
            
            cnt += 1
            
    print("训练完成 ✅")
    return model, tokenizer


# =======================
# 🧩 7. 主入口
# =======================
if __name__ == "__main__":
    # command args
    checkpoint_epoch = 450000
    ckpt_path = f"/home/qzj/A/RAG/Transformer/impl/checkpoints/transformer_epoch_{checkpoint_epoch}.pt"
    batch_size = 64
    acc_step = 16
    max_len = 512
    num_epochs = 10
    lr = 1.0
    warmup_steps = 6000
    log_step = 10000
    model, tokenizer = train_model(
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        accum_steps=acc_step,
        max_len=max_len,
        lr=lr,
        warmup_steps=warmup_steps,
        log_step=log_step,
        max_steps=450000 * 2,
        max_epoch=3
    )
    print("\n=== 最终推理示例 ===")
    for s in ["我喜欢你。", "你好，世界。", "你今天过得怎么样？"]:
        print(f"Input: {s}")
        print(f"Output: {greedy_decode(model, s, tokenizer)}")