# 🐟 FishAI Train v3

> FishLab-ai 自研 GPT 训练管线 — LLaMA-style 架构，从模型定义到量化导出

## v3 核心升级

| 特性 | v2 | v3 |
|------|----|----|
| 分词器 | 字节级 (260 vocab) | BPE (32K vocab) |
| 混合精度 | 无 | torch.amp (bf16/fp16) |
| 梯度累积 | 无 | 可配置 (默认 8 步) |
| 评估 | 无 | PPL + 6 个标准基准 |
| 检查点恢复 | 仅模型 | 模型 + 优化器 + 调度器 |
| 量化 | JSON + 假 FP16 | 二进制 + 真分组量化 |
| Flash Attention | 无 | scaled_dot_product_attention |
| KV Cache | 无 (全序列重计算) | 增量推理 |
| RoPE 缓冲区 | 每层重复 | 模型级共享 |

---

## 架构

```
                    FishAI v3 Transformer 架构
 ┌─────────────────────────────────────────────────────────────┐
 │  Input Token IDs  [B, T]                                    │
 │       │                                                      │
 │       ▼                                                      │
 │  ┌─────────────┐                                            │
 │  │ Token Embed  │  [B, T, d_model]                          │
 │  │ (无位置编码!) │                                            │
 │  └──────┬──────┘                                            │
 │         │                                                    │
 │    ┌────▼────────────────────────────────────┐              │
 │    │         Transformer Block × N            │              │
 │    │  ┌──────────────────────────────────┐   │              │
 │    │  │ x = x + GQA(RMSNorm(x))          │   │              │
 │    │  │        │                          │   │              │
 │    │  │   ┌────┴────┐                    │   │              │
 │    │  │   │ RoPE    │ ← 共享频率缓冲区   │   │              │
 │    │  │   │ Flash   │ ← SDPA 加速       │   │              │
 │    │  │   │ GQA     │ ← KV 头复用       │   │              │
 │    │  │   └────┬────┘                    │   │              │
 │    │  │        │                          │   │              │
 │    │  │ x = x + SwiGLU(RMSNorm(x))       │   │              │
 │    │  └──────────────────────────────────┘   │              │
 │    └─────────────────────────────────────────┘              │
 │         │                                                    │
 │    ┌────▼──────┐                                            │
 │    │ Final Norm │  RMSNorm                                  │
 │    └────┬──────┘                                            │
 │         │                                                    │
 │    ┌────▼──────┐                                            │
 │    │  LM Head   │  (权重绑定 = Token Embed)                 │
 │    └────┬──────┘                                            │
 │         │                                                    │
 │    Logits [B, T, vocab_size]                                │
 └─────────────────────────────────────────────────────────────┘
```

### 架构特性

- **RoPE** 旋转位置编码 — 零参数位置编码，支持外推
- **SwiGLU** 激活函数 — 三矩阵 FFN，表达力远超 GELU
- **RMSNorm** — 比 LayerNorm 更快更简
- **GQA** 分组查询注意力 — KV 头数 < Q 头数，省内存加速推理
- **权重绑定** — Token Embed 与 LM Head 共享 (小/中模型)
- **无偏置** — 现代发现 bias 在 RMSNorm+Residual 下冗余
- **Flash Attention** — 使用 `F.scaled_dot_product_attention` 加速
- **KV Cache** — 推理时增量计算，2-5× 加速

---

## 模型配置

| 参数 | Small (~34M) | Medium (~400M) | Large (~1.5B) |
|------|-------------|----------------|---------------|
| d_model | 512 | 896 | 1536 |
| n_heads (Q) | 8 | 14 | 12 |
| n_kv_heads | 4 | 2 | 4 |
| n_layers | 6 | 24 | 28 |
| d_ff (SwiGLU) | 1408 | 4864 | 8960 |
| max_seq_len | 2048 | 4096 | 4096 |
| weight_tying | ✓ | ✓ | ✗ |
| 参数量 | ~34M | ~400M | ~1.5B |
| 量化后 | ~12 MB | ~140 MB | ~520 MB |

---

## 项目结构

```
fishai-train/
├── model.py            # FishAI v3 模型 (RoPE/SwiGLU/RMSNorm/GQA/FlashAttn/KV Cache)
├── train.py            # 训练主脚本 (AMP/GradAccum/WandB/Checkpoint/评估)
├── quantize.py         # 混合精度量化导出 (分组量化/二进制格式/误差评估)
├── evaluate.py         # 基准测试框架 (PPL/MMLU/C-Eval/GSM8K/HumanEval/HellaSwag)
├── tokenizer_train.py  # BPE 分词器训练 (HuggingFace tokenizers/特殊 Token)
├── data_utils.py       # 数据处理工具 (过滤/去重/混合/二进制转换)
├── requirements.txt    # Python 依赖
└── README.md           # 文档
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 1. 训练 BPE 分词器

```bash
# 从训练数据训练 32K BPE 分词器
python tokenizer_train.py \
    --data data/train.txt \
    --vocab-size 32000 \
    --output-dir tokenizer
```

### 2. 数据处理

```bash
# 过滤数据
python data_utils.py filter \
    --input data/raw/ \
    --output data/filtered/ \
    --min-length 50

# 转换为二进制格式 (可选，加速训练)
python data_utils.py convert \
    --input data/filtered/ \
    --output data/train.bin \
    --tokenizer tokenizer/tokenizer.json

# 查看数据统计
python data_utils.py stats --input data/train.txt
```

### 3. 训练模型

```bash
# Small 模型 (单 GPU)
python train.py \
    --model-size small \
    --train-data data/train.txt \
    --val-data data/val.txt \
    --tokenizer tokenizer/tokenizer.json \
    --batch-size 16 \
    --grad-accum-steps 8 \
    --lr 6e-4 \
    --max-steps 100000 \
    --output-dir checkpoints/small

# Medium 模型 (多 GPU)
python train.py \
    --model-size medium \
    --train-data data/train.txt \
    --val-data data/val.txt \
    --tokenizer tokenizer/tokenizer.json \
    --batch-size 8 \
    --grad-accum-steps 16 \
    --lr 3e-4 \
    --max-steps 200000 \
    --amp-dtype bf16 \
    --output-dir checkpoints/medium \
    --wandb --wandb-project fishai

# 从检查点恢复训练
python train.py \
    --resume-from checkpoints/small/best_model.pt \
    --train-data data/train.txt \
    --tokenizer tokenizer/tokenizer.json
```

### 4. 量化导出

```bash
# 导出二进制格式 (推荐)
python quantize.py \
    --checkpoint checkpoints/small/best_model.pt \
    --output model_q4.bin \
    --format binary \
    --eval \
    --error-report

# 导出 JSON 格式 (兼容旧引擎)
python quantize.py \
    --checkpoint checkpoints/small/best_model.pt \
    --output model_q4.json \
    --format json
```

### 5. 评估

```bash
# 运行完整基准测试
python evaluate.py \
    --checkpoint checkpoints/small/best_model.pt \
    --tokenizer tokenizer/tokenizer.json \
    --val-data data/val.txt \
    --benchmarks ppl mmlu ceval gsm8k humaneval hellaswag \
    --output eval_results.json
```

---

## 训练超参数建议

### Small (~34M)

| 参数 | 值 |
|------|---|
| 学习率 | 6e-4 |
| 最小 LR | 6e-5 (10% of peak) |
| Warmup | 2000 steps |
| Batch Size | 16 × 8 = 128 (有效) |
| Weight Decay | 0.1 |
| 梯度裁剪 | 1.0 |
| 训练步数 | 100K-500K |
| 混合精度 | bf16 |
| 数据量 | 5-20B tokens |

### Medium (~400M)

| 参数 | 值 |
|------|---|
| 学习率 | 3e-4 |
| 最小 LR | 3e-5 |
| Warmup | 2000 steps |
| Batch Size | 8 × 16 = 128 |
| Weight Decay | 0.1 |
| 梯度裁剪 | 1.0 |
| 训练步数 | 200K-1M |
| 混合精度 | bf16 |
| 数据量 | 50-200B tokens |

### Large (~1.5B)

| 参数 | 值 |
|------|---|
| 学习率 | 1.5e-4 |
| 最小 LR | 1.5e-5 |
| Warmup | 4000 steps |
| Batch Size | 4 × 32 = 128 |
| Weight Decay | 0.1 |
| 梯度裁剪 | 1.0 |
| 训练步数 | 500K-2M |
| 混合精度 | bf16 |
| 数据量 | 200B-1T tokens |

---

## 评估方法论

### 基准测试

| 基准 | 方法 | 指标 | 描述 |
|------|------|------|------|
| PPL | 滑动窗口 | 困惑度 | 验证集语言模型质量 |
| MMLU | 5-shot | 准确率 | 57 个学科多选题 |
| C-Eval | 5-shot | 准确率 | 中文多学科评测 |
| GSM8K | 8-shot CoT | 准确率 | 小学数学推理 |
| HumanEval | 0-shot | pass@1, pass@10 | Python 代码生成 |
| HellaSwag | 0-shot | 准确率 | 常识推理补全 |

### 预期基准 (Small, ~34M)

| 基准 | 预期 | 说明 |
|------|------|------|
| PPL | 15-25 | 取决于训练数据量和质量 |
| MMLU | 25-30% | 小模型基线 |
| C-Eval | 25-30% | 中文基线 |
| GSM8K | 5-15% | CoT 推理对小模型困难 |
| HumanEval | 2-8% | 代码能力有限 |
| HellaSwag | 40-55% | 常识推理相对容易 |

---

## 量化策略

### 混合精度量化

| 层 | 精度 | 方式 | 原因 |
|----|------|------|------|
| Token Embedding | FP16 | 直接存储 | 精度敏感 |
| RMSNorm gamma | FP16 | 直接存储 | 精度敏感 |
| Q/K 投影 | INT8 | 逐通道 | 注意力精度敏感 |
| V/O 投影 | INT4 | 分组(128) | 对量化鲁棒 |
| FFN (gate/up/down) | INT4 | 分组(128) | 对量化鲁棒 |

### 分组量化 (GPTQ-style)

```
权重矩阵 W [out_features, in_features]
  → 展平
  → 按 128 元素分组
  → 每组独立计算 scale + zero_point
  → INT4 量化 (值域 [0, 15])
  → 打包 (2 个 4-bit → 1 个 uint8)
```

**分组量化 vs 逐通道量化:**
- 分组量化: 更精细的 scale，误差更小 (SNR 高 5-10 dB)
- 逐通道量化: 更少的 scale 参数，略大压缩率

### 量化感知评估

```bash
python quantize.py \
    --checkpoint checkpoints/best_model.pt \
    --output model_q4.bin \
    --eval --error-report
```

输出:
- 逐层 MSE、余弦相似度、SNR (dB)
- 量化前后 logits 差异
- Loss 差异

---

## 数据建议

### 数据混合比例

| 来源 | 比例 | 示例数据集 |
|------|------|-----------|
| 网页 | 50% | CommonCrawl, RefinedWeb |
| 代码 | 20% | The Stack, StarCoder |
| 书籍 | 10% | Books3 |
| 维基 | 10% | Wikipedia |
| 医学 | 5% | PubMed |
| 数学 | 5% | OpenMathInstruct |

### 数据质量

- **困惑度过滤**: 移除 PPL > 阈值的文档 (通常是噪声)
- **去重**: MinHash + LSH 近似去重
- **语言检测**: 确保中文/英文比例
- **长度过滤**: 50-100000 字符
- **特殊字符过滤**: 特殊字符比例 < 30%

### 数据量建议

| 模型 | 推荐 tokens | Chinchilla 最优 | 过度训练 |
|------|-----------|---------------|---------|
| Small (34M) | 5-20B | 0.7B | 20-100× |
| Medium (400M) | 50-200B | 8B | 10-50× |
| Large (1.5B) | 200B-1T | 30B | 5-30× |

---

## 路线图: 迈向 SOTA

### Phase 1: 基础设施 ✓
- [x] LLaMA-style 架构 (RoPE/SwiGLU/RMSNorm/GQA)
- [x] BPE 分词器 (32K)
- [x] 混合精度训练
- [x] 梯度累积
- [x] 检查点恢复
- [x] 量化导出

### Phase 2: 数据和训练
- [ ] 大规模数据收集 (100B+ tokens)
- [ ] 数据质量管线 (困惑度/去重/分类)
- [ ] 数据混合优化
- [ ] 长上下文训练 (8K-32K)
- [ ] 训练稳定性优化

### Phase 3: 对齐
- [ ] SFT (Supervised Fine-Tuning)
- [ ] RLHF (PPO/DPO)
- [ ] Chat template 标准化
- [ ] 安全性训练

### Phase 4: 高级特性
- [ ] MoE (Mixture of Experts)
- [ ] GQA → MQA → MLA
- [ ] 多模态 (视觉/音频)
- [ ] 推理优化 (Speculative Decoding, Quantization-Aware Training)
- [ ] 分布式训练 (FSDP/DeepSpeed)

### 预期里程碑

| 里程碑 | 模型 | 数据 | 预期 MMLU |
|--------|------|------|-----------|
| v3.0 | 34M | 5B tokens | 25-30% |
| v3.1 | 400M | 50B tokens | 40-50% |
| v3.2 | 1.5B | 200B tokens | 55-65% |
| v4.0 | 7B MoE | 1T tokens | 65-75% |

---

## 许可证

MIT License - FishLab-ai
