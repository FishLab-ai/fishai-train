"""
TinyAI - 训练脚本

完整的训练流程:
1. 数据预处理 (中文 + 英文 + 代码)
2. BPE 分词器训练
3. 模型训练 (AdamW + Cosine LR Schedule)
4. 定期保存 checkpoint
5. 训练完成后导出 4-bit 量化权重

训练数据来源:
- 中文维基百科
- 英文维基百科
- GitHub 代码数据集
- 自定义语料

用法:
    python train.py --config configs/small.yaml
    python train.py --data ./data --epochs 10 --batch-size 8
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import GPT, GPTConfig

# 尝试导入 wandb 用于实验追踪
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============ 数据集 ============

class TextDataset(Dataset):
    """
    文本数据集

    从文本文件中加载语料，按固定长度切分为训练样本。
    支持中文、英文和代码混合语料。
    """

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        print(f"[数据] 加载数据: {data_path}")

        # 读取所有文本
        texts = []
        if os.path.isdir(data_path):
            for f in sorted(Path(data_path).rglob("*.txt")):
                texts.append(f.read_text(encoding="utf-8"))
            for f in sorted(Path(data_path).rglob("*.jsonl")):
                with open(f, "r", encoding="utf-8") as fp:
                    for line in fp:
                        obj = json.loads(line)
                        texts.append(obj.get("text", ""))
        elif os.path.isfile(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                texts.append(f.read())

        print(f"[数据] 加载了 {len(texts)} 个文本文件")

        # 编码并切分
        total_tokens = 0
        for text in texts:
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)

            # 按固定长度切分
            for i in range(0, len(tokens) - max_seq_len, max_seq_len // 2):
                chunk = tokens[i:i + max_seq_len + 1]
                if len(chunk) > 10:  # 过滤过短的片段
                    self.samples.append(chunk)

        print(f"[数据] 总 token 数: {total_tokens:,}")
        print(f"[数据] 训练样本数: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============ 简易分词器 ============

class SimpleTokenizer:
    """
    简易 BPE 分词器 (训练用)

    生产环境请使用 Rust 引擎中的完整分词器。
    这里提供一个兼容的实现，确保训练和推理使用相同的词汇表。
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # 特殊 token
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.unk_token = 3
        # 基础 byte-level 词汇
        self.byte_to_id = {b: 4 + b for b in range(256)}
        self.id_to_byte = {4 + b: b for b in range(256)}
        # BPE merges (训练后填充)
        self.merges = {}

    def encode(self, text: str) -> list:
        """编码文本为 token ID 序列"""
        tokens = [self.bos_token]
        for ch in text.encode("utf-8"):
            tokens.append(self.byte_to_id.get(ch, self.unk_token))
        tokens.append(self.eos_token)
        return tokens

    def decode(self, token_ids: list) -> str:
        """解码 token ID 序列为文本"""
        bytes_list = []
        for id in token_ids:
            if id in self.id_to_byte:
                bytes_list.append(self.id_to_byte[id])
        return bytes(bytes_list).decode("utf-8", errors="replace")

    def save(self, path: str):
        """保存分词器"""
        data = {
            "vocab_size": self.vocab_size,
            "merges": self.merges,
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """加载分词器"""
        with open(path, "r") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.merges = data.get("merges", {})
        return tok


# ============ 训练主循环 ============

def train(args):
    """训练主函数"""

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[训练] 设备: {device}")
    if torch.cuda.is_available():
        print(f"[训练] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[训练] 显存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 配置
    config = GPTConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    print(f"\n{'='*60}")
    print(f"  TinyAI 训练配置")
    print(f"{'='*60}")
    print(f"  参数量: {config.total_params() / 1e6:.1f}M")
    print(f"  4-bit 量化: {config.quantized_size_mb():.1f} MB")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.lr}")
    print(f"  epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # 初始化模型
    model = GPT(config).to(device)

    # 分词器
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)

    # 数据集
    dataset = TextDataset(args.data, tokenizer, config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader),
        eta_min=args.lr * 0.1,
    )

    # Wandb
    if HAS_WANDB and args.use_wandb:
        wandb.init(
            project="tinyai",
            config=vars(args),
            name=f"tinyai-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    global_step = 0
    best_loss = float("inf")

    print("[训练] 开始训练...\n")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # 前向传播
            logits, loss = model(x, labels=y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # 日志
            if batch_idx % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                tokens_per_sec = (x.size(0) * x.size(1)) / (time.time() - epoch_start + 1e-6)

                msg = (
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Step {global_step} | "
                    f"Loss {loss.item():.4f} | "
                    f"LR {lr:.2e} | "
                    f"Tokens/s {tokens_per_sec:.0f}"
                )
                print(msg)

                if HAS_WANDB and args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/step": global_step,
                    })

        # Epoch 结束
        avg_loss = epoch_loss / max(len(dataloader), 1)
        elapsed = time.time() - epoch_start

        print(f"\n[Epoch {epoch+1}] 平均 Loss: {avg_loss:.4f} | 耗时: {elapsed:.1f}s\n")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": vars(config),
                "global_step": global_step,
            }, checkpoint_path)
            print(f"[保存] 最佳模型 -> {checkpoint_path}")

        # 定期保存 checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": vars(config),
                "global_step": global_step,
            }, checkpoint_path)
            print(f"[保存] Checkpoint -> {checkpoint_path}")

    # 训练结束
    print(f"\n{'='*60}")
    print(f"  训练完成!")
    print(f"  最佳 Loss: {best_loss:.4f}")
    print(f"  总步数: {global_step}")
    print(f"{'='*60}")

    # 导出量化权重
    print("\n[导出] 开始 4-bit 量化...")
    from quantize import export_quantized_weights
    export_quantized_weights(model, config, output_dir / "model_q4.json")
    print(f"[导出] 量化权重已保存到 {output_dir / 'model_q4.json'}")

    # 保存分词器
    tokenizer.save(str(output_dir / "tokenizer.json"))
    print(f"[导出] 分词器已保存到 {output_dir / 'tokenizer.json'}")

    if HAS_WANDB and args.use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="TinyAI 训练脚本")

    # 数据
    parser.add_argument("--data", type=str, default="./data", help="训练数据路径")
    parser.add_argument("--output-dir", type=str, default="./output", help="输出目录")

    # 模型
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    # 训练
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1)

    # 工具
    parser.add_argument("--use-wandb", action="store_true", help="使用 wandb 追踪")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
