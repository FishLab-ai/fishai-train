"""
FishAI v2 训练管线 — 小体积最聪明

训练升级:
- 更长 warmup (1-5% steps)
- 余弦退火到 min_lr (10% of peak)
- AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1)
- 梯度裁剪 1.0
- 过度训练策略 (Chinchilla 的 20× → 100-500×)
- 数据混合: 50-60% web + 15-20% code + 10-15% books
"""

import os
import math
import json
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import GPT, GPTConfig


# ──────────────── 数据集 ────────────────

class TextDataset(Dataset):
    """文本数据集: 滑动窗口切分"""

    def __init__(self, text: str, tokenizer, max_seq_len: int = 512, stride: int = 256):
        self.token_ids = tokenizer.encode(text)
        self.max_seq_len = max_seq_len
        self.stride = stride

        # 滑动窗口切分
        self.chunks = []
        for i in range(0, len(self.token_ids) - max_seq_len, stride):
            self.chunks.append(self.token_ids[i:i + max_seq_len + 1])

        if len(self.chunks) == 0 and len(self.token_ids) > 1:
            self.chunks.append(self.token_ids[:min(len(self.token_ids), max_seq_len + 1)])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        # Pad to max_seq_len
        if x.size(0) < self.max_seq_len:
            pad_len = self.max_seq_len - x.size(0)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])

        return x, y


class SimpleTokenizer:
    """简单 byte-level 分词器 (与 Rust 引擎兼容)"""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def encode(self, text: str) -> list:
        tokens = [1]  # BOS
        for ch in text:
            buf = [0] * 4
            s = ch.encode('utf-8')
            for b in s:
                tokens.append(4 + b)
        tokens.append(2)  # EOS
        return tokens

    def decode(self, token_ids: list) -> str:
        bytes_list = []
        for tid in token_ids:
            if tid <= 3:
                continue
            if 4 <= tid <= 259:
                bytes_list.append(tid - 4)
        return bytes(bytes_list).decode('utf-8', errors='replace')


# ──────────────── 训练配置 ────────────────

@dataclass
class TrainConfig:
    # 数据
    data_path: str = "data/train.txt"
    vocab_size: int = 32000
    max_seq_len: int = 512

    # 模型
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 4       # GQA
    n_layers: int = 6
    d_ff: int = 1408           # SwiGLU
    weight_tying: bool = True

    # 训练
    batch_size: int = 16
    learning_rate: float = 6e-4
    min_lr: float = 6e-5       # 10% of peak
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 50000
    save_every: int = 5000
    eval_every: int = 1000

    # 输出
    output_dir: str = "checkpoints"


# ──────────────── 训练循环 ────────────────

def train(config: TrainConfig):
    print(f"\n{'='*60}")
    print(f"  FishAI v2 — 训练管线")
    print(f"{'='*60}\n")

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 模型
    model_config = GPTConfig(
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        weight_tying=config.weight_tying,
    )
    model = GPT(model_config)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"[设备] {device}")

    # 数据
    tokenizer = SimpleTokenizer(config.vocab_size)

    if os.path.exists(config.data_path):
        with open(config.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"[数据] 加载 {len(text)} 字符")
    else:
        # 生成示例数据
        print(f"[数据] 未找到 {config.data_path}, 使用示例数据")
        text = "FishAI 是 FishLab-ai 团队自研的 AI 助手。" * 1000

    dataset = TextDataset(text, tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    print(f"[数据] {len(dataset)} 个训练样本")

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # 学习率调度: warmup + cosine decay
    def get_lr(step):
        if step < config.warmup_steps:
            return config.learning_rate * step / config.warmup_steps
        decay_steps = config.max_steps - config.warmup_steps
        progress = (step - config.warmup_steps) / decay_steps
        return config.min_lr + 0.5 * (config.learning_rate - config.min_lr) * (1 + math.cos(math.pi * progress))

    # 训练循环
    model.train()
    global_step = 0
    best_loss = float('inf')
    data_iter = iter(dataloader)

    print(f"\n[训练] 开始 — max_steps={config.max_steps}, warmup={config.warmup_steps}")
    print(f"[训练] LR: {config.learning_rate} -> {config.min_lr} (cosine)")
    print()

    start_time = time.time()

    while global_step < config.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x, y = batch
        x, y = x.to(device), y.to(device)

        # 前向传播
        logits, loss = model(x, labels=y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # 学习率调整
        lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()

        global_step += 1

        # 日志
        if global_step % 100 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = global_step * config.batch_size * config.max_seq_len / elapsed
            print(f"[Step {global_step}] loss={loss.item():.4f} lr={lr:.6f} "
                  f"tok/s={tokens_per_sec:.0f}")

        # 保存检查点
        if global_step % config.save_every == 0:
            checkpoint = {
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'config': model_config,
            }
            path = os.path.join(config.output_dir, f"checkpoint_{global_step}.pt")
            torch.save(checkpoint, path)
            print(f"[保存] 检查点 -> {path}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_path = os.path.join(config.output_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                print(f"[保存] 最佳模型 -> {best_path}")

    # 训练结束，导出量化权重
    print(f"\n{'='*60}")
    print(f"  训练完成! 最佳 loss: {best_loss:.4f}")
    print(f"{'='*60}")

    # 导出量化权重
    from quantize import export_quantized_weights
    export_path = os.path.join(config.output_dir, "model_q4.json")
    export_quantized_weights(model, model_config, export_path)
    print(f"[导出] 量化权重 -> {export_path}")


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
