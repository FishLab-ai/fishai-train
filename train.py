"""
FishAI v3 训练管线 — 完整的 LLM 训练框架

v3 训练升级:
- BPE 分词器 (HuggingFace tokenizers, 32K 词表)
- 混合精度训练 (torch.amp autocast + GradScaler)
- 梯度累积 (可配置, 默认 8 步 → 有效 batch 128)
- 学习率调度 (LambdaLR with warmup + cosine decay)
- 评估: 验证集 loss + 生成样本文本
- 检查点恢复: 保存/加载 optimizer, scheduler, step, best_loss
- 权重衰减排除 (embedding 和 norm 参数不衰减)
- Wandb 日志集成
- DataLoader with epoch tracking
- 数据管线: 支持 txt/jsonl, 数据混合比例
- 命令行参数解析 (argparse)
"""

import os
import sys
import math
import json
import time
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import GPT, GPTConfig, get_model_config, save_model, load_model


# ──────────────── 日志配置 ────────────────

def setup_logging(log_dir: str, name: str = "fishai_train") -> logging.Logger:
    """配置训练日志"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    # 文件输出
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "train.log"), encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ──────────────── 数据集 ────────────────

class TextDataset(Dataset):
    """
    文本数据集: 滑动窗口切分
    支持 BPE tokenizer 输出
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        stride: Optional[int] = None,
        text_key: str = "text",
    ):
        """
        Args:
            data_path: 数据文件路径 (txt/jsonl)
            tokenizer: FishAITokenizer 实例
            max_seq_len: 最大序列长度
            stride: 滑动窗口步长 (None = max_seq_len)
            text_key: JSONL 文本字段名
        """
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len // 2
        self.tokenizer = tokenizer

        # 加载和分词
        print(f"[数据集] 加载: {data_path}")
        documents = self._load_documents(data_path, text_key)
        print(f"[数据集] 加载 {len(documents)} 个文档")

        # 分词所有文档
        all_token_ids = []
        for doc in documents:
            ids = tokenizer.encode(doc, add_bos=True, add_eos=True)
            all_token_ids.extend(ids)

        print(f"[数据集] 总 token 数: {len(all_token_ids):,}")

        # 滑动窗口切分
        self.chunks = []
        for i in range(0, len(all_token_ids) - max_seq_len, self.stride):
            self.chunks.append(all_token_ids[i:i + max_seq_len + 1])

        # 处理最后一段不足 max_seq_len 的情况
        if len(all_token_ids) > max_seq_len + 1:
            last = all_token_ids[-(max_seq_len + 1):]
            if last not in self.chunks:
                self.chunks.append(last)
        elif len(all_token_ids) > 1:
            self.chunks.append(all_token_ids)

        print(f"[数据集] 切分为 {len(self.chunks)} 个训练片段 "
              f"(max_seq_len={max_seq_len}, stride={self.stride})")

    def _load_documents(self, path: str, text_key: str) -> List[str]:
        """加载文档"""
        ext = Path(path).suffix
        documents = []

        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get(text_key, "")
                        if text:
                            documents.append(text)
                    except json.JSONDecodeError:
                        continue
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            # 按段落分割
            for para in text.split("\n\n"):
                para = para.strip()
                if para:
                    documents.append(para)

        return documents

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        # Pad to max_seq_len
        if x.size(0) < self.max_seq_len:
            pad_len = self.max_seq_len - x.size(0)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])

        return x, y


class BinaryDataset(Dataset):
    """
    二进制格式数据集 (由 data_utils.py 生成)
    支持更快的加载速度
    """

    def __init__(self, data_path: str, max_seq_len: int = 2048):
        import struct
        self.max_seq_len = max_seq_len

        with open(data_path, "rb") as f:
            # 读取头部
            magic = f.read(4)
            if magic != b"FAID":
                raise ValueError(f"无效的二进制文件: {magic}")

            version = struct.unpack("<I", f.read(4))[0]
            vocab_size = struct.unpack("<I", f.read(4))[0]
            n_chunks = struct.unpack("<I", f.read(4))[0]
            file_max_seq_len = struct.unpack("<I", f.read(4))[0]

            # 读取源名称
            n_sources = struct.unpack("<I", f.read(4))[0]
            self.sources = []
            for _ in range(n_sources):
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                self.sources.append(name)

            # 读取索引
            self.index = []
            for i in range(n_chunks):
                offset = struct.unpack("<Q", f.read(8))[0]
                length = struct.unpack("<I", f.read(4))[0]
                source_id = struct.unpack("<I", f.read(4))[0]
                self.index.append((offset, length, source_id))

            # 读取所有数据
            self.data_start = f.tell()
            self.data = f.read()

        self.n_chunks = n_chunks
        print(f"[二进制数据集] 加载: {n_chunks} 个片段, "
              f"vocab_size={vocab_size}, max_seq_len={file_max_seq_len}")

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import struct
        offset, length, source_id = self.index[idx]

        # 读取 token IDs
        token_data = self.data[offset:offset + length * 4]
        token_ids = list(struct.unpack(f"<{length}I", token_data))

        x = torch.tensor(token_ids[:-1], dtype=torch.long)
        y = torch.tensor(token_ids[1:], dtype=torch.long)

        # Pad
        if x.size(0) < self.max_seq_len:
            pad_len = self.max_seq_len - x.size(0)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -1, dtype=torch.long)])
        elif x.size(0) > self.max_seq_len:
            x = x[:self.max_seq_len]
            y = y[:self.max_seq_len]

        return x, y


# ──────────────── 学习率调度 ────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    创建带 warmup 的余弦退火学习率调度器

    Args:
        optimizer: 优化器
        warmup_steps: warmup 步数
        max_steps: 最大训练步数
        min_lr_ratio: 最小学习率与峰值学习率的比例

    Returns:
        LambdaLR 调度器
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # 线性 warmup
            return step / max(warmup_steps, 1)
        # 余弦退火
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ──────────────── 权重衰减排除 ────────────────

def separate_weight_decay_params(
    model: nn.Module,
    weight_decay: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    分离需要/不需要权重衰减的参数

    规则:
    - embedding 参数: 不衰减
    - norm (RMSNorm gamma) 参数: 不衰减
    - bias 参数: 不衰减 (虽然我们的模型没有 bias)
    - 其他参数: 衰减

    Returns:
        参数组列表 [{"params": [...], "weight_decay": ...}, ...]
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if ("embedding" in name or "norm" in name or "gamma" in name or
                "bias" in name):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"[优化器] 权重衰减参数: {n_decay / 1e6:.1f}M, "
          f"不衰减参数: {n_no_decay / 1e6:.1f}M")

    return param_groups


# ──────────────── 评估 ────────────────

@torch.no_grad()
def evaluate(
    model: GPT,
    eval_dataloader: DataLoader,
    device: torch.device,
    max_steps: Optional[int] = None,
    desc: str = "验证",
) -> Dict[str, float]:
    """
    评估模型

    Args:
        model: 模型
        eval_dataloader: 验证数据加载器
        device: 设备
        max_steps: 最大评估步数 (None = 全部)
        desc: 描述

    Returns:
        {"loss": ..., "ppl": ...}
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch_idx, (x, y) in enumerate(eval_dataloader):
        if max_steps is not None and batch_idx >= max_steps:
            break

        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, labels=y)

        if loss is not None:
            # 计算非 padding token 数量
            n_tokens = (y != -1).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            n_batches += 1

    model.train()

    if total_tokens == 0 or n_batches == 0:
        return {"loss": float("inf"), "ppl": float("inf")}

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))  # 防止溢出

    return {"loss": avg_loss, "ppl": ppl}


@torch.no_grad()
def generate_sample(
    model: GPT,
    tokenizer,
    device: torch.device,
    prompt: str = "FishAI 是",
    max_new_tokens: int = 64,
    temperature: float = 0.8,
) -> str:
    """
    生成样本文本

    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备
        prompt: 提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度

    Returns:
        生成的文本
    """
    model.eval()

    input_ids = tokenizer.encode(prompt, add_bos=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=50,
        eos_token_id=tokenizer.eos_id,
    )

    output_ids = generated[0].tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    model.train()
    return text


# ──────────────── Wandb 日志 ────────────────

class WandbLogger:
    """Wandb 日志记录器"""

    def __init__(
        self,
        project: str = "fishai",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._run = None

        if enabled:
            try:
                import wandb
                self._wandb = wandb
                self._run = wandb.init(
                    project=project,
                    name=name,
                    config=config,
                )
                print(f"[Wandb] 已启动: project={project}, name={name}")
            except ImportError:
                print("[Wandb] wandb 未安装，跳过日志记录")
                self.enabled = False
            except Exception as e:
                print(f"[Wandb] 启动失败: {e}")
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """记录指标"""
        if self.enabled and self._run is not None:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        """结束记录"""
        if self.enabled and self._run is not None:
            self._wandb.finish()


# ──────────────── 训练配置 ────────────────

@dataclass
class TrainConfig:
    """训练配置"""
    # 模型
    model_size: str = "small"        # small / medium / large
    vocab_size: int = 32000
    max_seq_len: int = 2048

    # 数据
    train_data: str = "data/train.txt"
    val_data: Optional[str] = None   # 验证数据路径
    tokenizer_path: Optional[str] = None  # 分词器路径

    # 训练
    batch_size: int = 16             # 每个 GPU 的 batch size
    grad_accum_steps: int = 8        # 梯度累积步数 (有效 batch = 16 * 8 = 128)
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1        # 最小学习率 = learning_rate * min_lr_ratio
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000

    # 混合精度
    use_amp: bool = True             # 是否使用混合精度
    amp_dtype: str = "bf16"          # fp16 / bf16

    # 评估和保存
    eval_interval: int = 1000        # 评估间隔步数
    eval_max_steps: int = 100        # 评估最大步数
    save_interval: int = 5000        # 保存间隔步数
    generate_interval: int = 2000    # 生成样本间隔步数
    generate_prompt: str = "FishAI 是"  # 生成样本提示

    # 检查点恢复
    resume_from: Optional[str] = None  # 检查点路径

    # 输出
    output_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Wandb
    wandb_project: str = "fishai"
    wandb_name: Optional[str] = None
    use_wandb: bool = False

    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True

    def get_effective_batch_size(self) -> int:
        """获取有效 batch size"""
        return self.batch_size * self.grad_accum_steps

    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return self.__dict__.copy()


# ──────────────── 训练主函数 ────────────────

def train(config: TrainConfig) -> None:
    """
    训练主函数

    完整的训练管线:
    1. 加载分词器和数据
    2. 创建模型和优化器
    3. 混合精度训练循环
    4. 评估和保存
    """
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 日志
    logger = setup_logging(config.log_dir)
    logger.info(f"FishAI v3 训练开始")
    logger.info(f"配置: {json.dumps(config.to_dict(), indent=2, default=str)}")

    # ── 设备 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU 内存: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── 混合精度 ──
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16
    use_amp = config.use_amp and torch.cuda.is_available()
    # bf16 不需要 GradScaler
    use_scaler = use_amp and amp_dtype == torch.float16

    logger.info(f"混合精度: {'启用' if use_amp else '禁用'} "
                f"(dtype={config.amp_dtype}, scaler={use_scaler})")

    # ── 分词器 ──
    from tokenizer_train import FishAITokenizer
    if config.tokenizer_path and os.path.exists(config.tokenizer_path):
        tokenizer = FishAITokenizer(config.tokenizer_path)
    else:
        # 使用默认字节级 BPE
        logger.warning("未找到分词器文件，使用默认字节级 BPE (建议先运行 tokenizer_train.py)")
        tokenizer = FishAITokenizer()

    logger.info(f"分词器: vocab_size={tokenizer.vocab_size}")

    # ── 模型 ──
    model_config = get_model_config(config.model_size)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_len = config.max_seq_len

    # 检查点恢复
    global_step = 0
    best_loss = float("inf")
    optimizer_state = None
    scheduler_state = None

    if config.resume_from and os.path.exists(config.resume_from):
        model, optimizer_state, scheduler_state, global_step, best_loss = load_model(
            config.resume_from, device=device
        )
        # 更新 model_config
        model_config = model.config
        logger.info(f"从检查点恢复: step={global_step}, best_loss={best_loss:.4f}")
    else:
        model = GPT(model_config)
        model = model.to(device)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: {total_params / 1e6:.1f}M (可训练: {trainable_params / 1e6:.1f}M)")

    # ── 数据 ──
    logger.info(f"加载训练数据: {config.train_data}")
    train_dataset = TextDataset(
        config.train_data, tokenizer,
        max_seq_len=config.max_seq_len,
    )

    val_dataset = None
    if config.val_data and os.path.exists(config.val_data):
        logger.info(f"加载验证数据: {config.val_data}")
        val_dataset = TextDataset(
            config.val_data, tokenizer,
            max_seq_len=config.max_seq_len,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

    # ── 优化器 ──
    param_groups = separate_weight_decay_params(model, config.weight_decay)
    optimizer = AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        logger.info("已加载优化器状态")

    # ── 学习率调度 ──
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        min_lr_ratio=config.min_lr_ratio,
    )

    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
        logger.info("已加载调度器状态")

    # ── GradScaler ──
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── Wandb ──
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        name=config.wandb_name or f"fishai-{config.model_size}",
        config=config.to_dict(),
        enabled=config.use_wandb,
    )

    # ── 训练循环 ──
    logger.info(f"\n{'='*60}")
    logger.info(f"  训练开始")
    logger.info(f"  有效 batch: {config.get_effective_batch_size()} "
                f"({config.batch_size} x {config.grad_accum_steps})")
    logger.info(f"  学习率: {config.learning_rate} → "
                f"{config.learning_rate * config.min_lr_ratio} (cosine)")
    logger.info(f"  Warmup: {config.warmup_steps} steps")
    logger.info(f"  最大步数: {config.max_steps}")
    logger.info(f"  评估间隔: {config.eval_interval} steps")
    logger.info(f"  保存间隔: {config.save_interval} steps")
    logger.info(f"{'='*60}\n")

    model.train()
    data_iter = iter(train_dataloader)
    start_time = time.time()
    log_loss = 0.0
    log_tokens = 0

    while global_step < config.max_steps:
        # 梯度累积循环
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for micro_step in range(config.grad_accum_steps):
            # 获取数据
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # 前向传播 (混合精度)
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits, loss, _ = model(x, labels=y)

            # 缩放 loss 以适应梯度累积
            scaled_loss = loss / config.grad_accum_steps

            # 反向传播
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accumulated_loss += loss.item()

            # 统计
            n_tokens = (y != -1).sum().item()
            log_loss += loss.item() * n_tokens
            log_tokens += n_tokens

        # 梯度裁剪
        if use_scaler:
            scaler.unscale_(optimizer)
        if config.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # 优化器步进
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()
        global_step += 1

        # ── 日志 ──
        if global_step % 100 == 0:
            elapsed = time.time() - start_time
            avg_loss = log_loss / max(log_tokens, 1)
            tokens_per_sec = log_tokens / elapsed
            lr = scheduler.get_last_lr()[0]

            logger.info(
                f"[Step {global_step}/{config.max_steps}] "
                f"loss={avg_loss:.4f} lr={lr:.6f} "
                f"tok/s={tokens_per_sec:.0f} "
                f"accum_loss={accumulated_loss:.4f}"
            )

            wandb_logger.log({
                "train/loss": avg_loss,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_sec,
                "train/step": global_step,
            }, step=global_step)

            log_loss = 0.0
            log_tokens = 0
            start_time = time.time()

        # ── 评估 ──
        if val_dataloader is not None and global_step % config.eval_interval == 0:
            eval_results = evaluate(
                model, val_dataloader, device,
                max_steps=config.eval_max_steps,
            )
            logger.info(
                f"[评估 Step {global_step}] "
                f"val_loss={eval_results['loss']:.4f} "
                f"val_ppl={eval_results['ppl']:.2f}"
            )
            wandb_logger.log({
                "eval/loss": eval_results["loss"],
                "eval/ppl": eval_results["ppl"],
            }, step=global_step)

        # ── 生成样本 ──
        if global_step % config.generate_interval == 0:
            sample_text = generate_sample(
                model, tokenizer, device,
                prompt=config.generate_prompt,
                max_new_tokens=64,
            )
            logger.info(f"[生成 Step {global_step}] {sample_text[:200]}")
            wandb_logger.log({
                "generate/sample": sample_text,
            }, step=global_step)

        # ── 保存检查点 ──
        if global_step % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.output_dir, f"checkpoint_{global_step}.pt"
            )
            save_model(
                model, checkpoint_path,
                optimizer=optimizer,
                scheduler=scheduler,
                step=global_step,
                best_loss=best_loss,
            )

            # 保存最佳模型
            current_loss = accumulated_loss / config.grad_accum_steps
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = os.path.join(config.output_dir, "best_model.pt")
                save_model(
                    model, best_path,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    best_loss=best_loss,
                )
                logger.info(f"[保存] 最佳模型 (loss={best_loss:.4f}) -> {best_path}")

    # ── 训练结束 ──
    logger.info(f"\n{'='*60}")
    logger.info(f"  训练完成!")
    logger.info(f"  总步数: {global_step}")
    logger.info(f"  最佳 loss: {best_loss:.4f}")
    logger.info(f"{'='*60}")

    # 保存最终模型
    final_path = os.path.join(config.output_dir, "final_model.pt")
    save_model(
        model, final_path,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        best_loss=best_loss,
    )

    # 导出量化权重
    try:
        from quantize import export_quantized_weights
        export_path = os.path.join(config.output_dir, "model_q4.bin")
        export_quantized_weights(model, model_config, export_path)
        logger.info(f"[导出] 量化权重 -> {export_path}")
    except Exception as e:
        logger.warning(f"[导出] 量化权重失败: {e}")

    wandb_logger.finish()


# ──────────────── 命令行入口 ────────────────

def parse_args() -> TrainConfig:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="FishAI v3 训练")

    # 模型
    parser.add_argument("--model-size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="模型大小 (默认: small)")
    parser.add_argument("--vocab-size", type=int, default=32000,
                        help="词表大小")
    parser.add_argument("--max-seq-len", type=int, default=2048,
                        help="最大序列长度")

    # 数据
    parser.add_argument("--train-data", type=str, required=True,
                        help="训练数据路径 (txt/jsonl)")
    parser.add_argument("--val-data", type=str, default=None,
                        help="验证数据路径")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="分词器路径")

    # 训练
    parser.add_argument("--batch-size", type=int, default=16,
                        help="每个 GPU 的 batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=6e-4,
                        help="学习率")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1,
                        help="最小学习率比例")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="权重衰减")
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="Warmup 步数")
    parser.add_argument("--max-steps", type=int, default=100000,
                        help="最大训练步数")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪阈值")

    # 混合精度
    parser.add_argument("--no-amp", action="store_true",
                        help="禁用混合精度")
    parser.add_argument("--amp-dtype", type=str, default="bf16",
                        choices=["fp16", "bf16"],
                        help="混合精度数据类型")

    # 评估和保存
    parser.add_argument("--eval-interval", type=int, default=1000,
                        help="评估间隔步数")
    parser.add_argument("--save-interval", type=int, default=5000,
                        help="保存间隔步数")
    parser.add_argument("--generate-interval", type=int, default=2000,
                        help="生成样本间隔步数")

    # 恢复
    parser.add_argument("--resume-from", type=str, default=None,
                        help="从检查点恢复训练")

    # 输出
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="输出目录")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="日志目录")

    # Wandb
    parser.add_argument("--wandb", action="store_true",
                        help="启用 Wandb 日志")
    parser.add_argument("--wandb-project", type=str, default="fishai",
                        help="Wandb 项目名")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Wandb 运行名")

    # 数据加载
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader 工作进程数")

    args = parser.parse_args()

    config = TrainConfig(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        train_data=args.train_data,
        val_data=args.val_data,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        generate_interval=args.generate_interval,
        resume_from=args.resume_from,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        num_workers=args.num_workers,
    )

    return config


if __name__ == "__main__":
    config = parse_args()
    train(config)
