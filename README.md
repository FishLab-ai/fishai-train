<div align="center">

# 🧠 FishAI Train

**Python 训练管线 — 从零训练小体积最聪明的 LLM**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/FishLab-ai/fishai-train)
[![Version](https://img.shields.io/badge/version-v3.1.0-blue)](https://github.com/FishLab-ai/fishai-train)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-yellow)](https://www.python.org/)

**Python Training Pipeline for Small yet Smart LLMs**

[English](#english) | [中文](#中文)

</div>

---

## 中文

### 项目简介

FishAI Train 是 FishLab-ai 自研的 LLM 训练管线，采用 LLaMA-3/Phi-3 同代架构（RoPE + SwiGLU + RMSNorm + GQA + Weight Tying + Flash Attention），提供从分词器训练到量化导出的完整工具链。目标：在极小参数量下达到或超越 Pythia-70M 的 WikiText-103 困惑度。

### 核心特性

- [x] **BPE 分词器** — 基于 HuggingFace tokenizers，32K 词表，支持 chat template
- [x] **混合精度训练** — torch.amp autocast + GradScaler
- [x] **梯度累积** — 可配置步数（默认 8 步 → 有效 batch 128）
- [x] **余弦学习率调度** — Warmup + Cosine Decay + Min LR
- [x] **权重衰减排除** — Embedding 和 Norm 参数不衰减
- [x] **检查点恢复** — 保存/加载 optimizer + scheduler + step + best_loss
- [x] **Wandb 日志集成** — 实验追踪
- [x] **混合精度量化导出** — INT4 group128 + INT8 + FP16

### 模型架构

| 组件 | 实现 | 说明 |
|------|------|------|
| 位置编码 | RoPE (Rotary Position Embedding) | 零参数位置编码，共享频率缓冲区 |
| 激活函数 | SwiGLU | 比 GELU 更强表达力，gate + up + down 三投影 |
| 归一化 | RMSNorm | 比 LayerNorm 更快更简 |
| 注意力 | GQA (Grouped Query Attention) | 省 KV 缓存，加速推理 |
| 权重绑定 | Weight Tying | Token Embedding 与 LM Head 共享 |
| 注意力计算 | Flash Attention | `torch.nn.functional.scaled_dot_product_attention` |
| 偏置 | No Bias | 现代发现 bias 在 RMSNorm 下冗余 |
| 推理加速 | KV Cache | 增量推理，避免重复计算 |

### 模型尺寸

| 配置 | 参数量 | d_model | n_layers | n_heads | n_kv_heads | d_ff | FP32 大小 | 4-bit 大小 |
|------|--------|---------|----------|---------|------------|------|-----------|------------|
| **small** | ~34M | 512 | 6 | 8 | 4 | 1,408 | ~130 MB | ~17 MB |
| **medium** | ~400M | 1,024 | 12 | 16 | 4 | 2,816 | ~1.5 GB | ~200 MB |
| **large** | ~1.5B | 2,048 | 24 | 32 | 8 | 5,632 | ~5.6 GB | ~750 MB |

> small 配置使用 GQA (8 Q heads / 4 KV heads)，权重绑定后 ~34M 参数，4-bit 量化仅约 17MB。

### 量化策略

| 层类型 | 量化方式 | 说明 |
|--------|----------|------|
| Token Embedding / RMSNorm gamma | **FP16** | 精度敏感层保留高精度 |
| Q/K 投影 | **INT8** | 注意力精度更敏感 |
| V/O 投影 / FFN 权重 | **INT4 group128** | 对量化鲁棒，128 元素一组独立 scale/zero_point |

> 预期：3-4× 压缩率，困惑度损失 < 1%

### 对标基准

| 模型 | 参数量 | WikiText-103 PPL | WikiText-2 PPL | 训练数据 |
|------|--------|-------------------|----------------|----------|
| **FishAI-Small (目标)** | ~34M | ≤ 56.0 | ≤ 42.0 | — |
| Pythia-70M | 70M | 56.0 | 42.0 | 300B tokens |
| Pythia-160M | 160M | 36.8 | 27.0 | 300B tokens |
| GPT-2 Small | 124M | 37.5 | 29.0 | ~8B tokens |
| SmolLM2-135M | 135M | ~42.0 | ~32.0 | 2T tokens |

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 快速训练 FishAI-Small（WikiText-103，5000 步）
python quick_train.py --config small --max-steps 5000 --device cuda

# 完整训练（支持更多配置）
python train.py \
  --config small \
  --output-dir checkpoints \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 5e-4 \
  --warmup 100 \
  --max-steps 10000 \
  --seq-len 512 \
  --device cuda

# 训练 BPE 分词器
python tokenizer_train.py \
  --data train_data.txt \
  --vocab-size 32000 \
  --output-dir tokenizer

# 量化导出
python quantize.py \
  --model checkpoints/fishai-best.pt \
  --output fishai-small-int4.bin \
  --strategy mixed

# 运行基准测试
python benchmark.py self-test --config small --device cuda
python benchmark.py full --model checkpoints/fishai-best.pt --device cuda

# 对标跑分（HuggingFace 预训练模型）
python reference_benchmark.py

# 评估（MMLU / HellaSwag / GSM8K 等）
python evaluate.py --model checkpoints/fishai-best.pt --benchmark mmlu
```

### 文件说明

| 文件 | 说明 |
|------|------|
| `model.py` | 模型定义 — GPT (RoPE + SwiGLU + RMSNorm + GQA + Weight Tying + Flash Attention + KV Cache) |
| `train.py` | 完整训练管线 — 混合精度 + 梯度累积 + 余弦 LR + 检查点 + Wandb |
| `quick_train.py` | 快速训练脚本 — 训练 + 跑分一体化，未达标自动调整 |
| `tokenizer_train.py` | BPE 分词器训练 — HuggingFace tokenizers + 特殊 token + chat template |
| `quantize.py` | 混合精度量化 — INT4 group128 + INT8 + FP16 + 二进制导出 |
| `benchmark.py` | 标准基准测试 — WikiText-103/2 PPL + Penn Treebank + 生成质量 |
| `reference_benchmark.py` | 对标跑分 — Pythia-70M/160M + GPT-2 Small 的 WikiText PPL |
| `evaluate.py` | 评估框架 — MMLU / C-Eval / GSM8K / HumanEval / HellaSwag |
| `data_utils.py` | 数据处理 — 质量过滤 + 去重 + 数据混合 + 二进制格式转换 |
| `requirements.txt` | Python 依赖 |

### 数据管线

```
原始数据 (txt/jsonl)
    │
    ├──→ DataFilter (长度/去重/语言/特殊字符/数字比例)
    │
    ├──→ DataMixer (web 50% + code 20% + books 10% + wiki 10% + medical 5% + math 5%)
    │
    ├──→ FishAITokenizer (BPE 编码)
    │
    └──→ 二进制格式 (.bin) — 滑动窗口切分，快速训练加载
```

---

## English

### Overview

FishAI Train is FishLab-ai's LLM training pipeline, using a LLaMA-3/Phi-3 era architecture (RoPE + SwiGLU + RMSNorm + GQA + Weight Tying + Flash Attention). It provides a complete toolchain from tokenizer training to quantized export. Goal: match or exceed Pythia-70M's WikiText-103 perplexity at a fraction of the parameters.

### Key Features

- [x] **BPE Tokenizer** — HuggingFace tokenizers, 32K vocab, chat template support
- [x] **Mixed Precision Training** — torch.amp autocast + GradScaler
- [x] **Gradient Accumulation** — Configurable steps (default 8 → effective batch 128)
- [x] **Cosine LR Schedule** — Warmup + Cosine Decay + Min LR ratio
- [x] **Weight Decay Exclusion** — Embedding and Norm params excluded from decay
- [x] **Checkpoint Resume** — Save/load optimizer + scheduler + step + best_loss
- [x] **Wandb Integration** — Experiment tracking
- [x] **Mixed-Precision Quantization** — INT4 group128 + INT8 + FP16 export

### Quick Start

```bash
pip install -r requirements.txt

# Quick train FishAI-Small
python quick_train.py --config small --max-steps 5000 --device cuda

# Full training
python train.py --config small --max-steps 10000 --device cuda

# Train BPE tokenizer
python tokenizer_train.py --data train_data.txt --vocab-size 32000

# Quantize model
python quantize.py --model checkpoints/fishai-best.pt --output fishai-small-int4.bin

# Run benchmarks
python benchmark.py self-test --config small
python benchmark.py full --model checkpoints/fishai-best.pt

# Evaluate
python evaluate.py --model checkpoints/fishai-best.pt --benchmark mmlu
```

### Benchmark Targets

| Model | Params | WikiText-103 PPL |
|-------|--------|-------------------|
| **FishAI-Small (target)** | ~34M | ≤ 56.0 |
| Pythia-70M | 70M | 56.0 |
| Pythia-160M | 160M | 36.8 |
| GPT-2 Small | 124M | 37.5 |
| SmolLM2-135M | 135M | ~42.0 |

### License

MIT License - FishLab-ai

---

<div align="center">

**FishAI Train v3.1.0** — Made with 🧠 by [FishLab-ai](https://github.com/FishLab-ai)

</div>
