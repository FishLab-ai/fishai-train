# 🐟 FishAI Train v3

> FishLab-ai 训练管线 — PyTorch，从分词器训练到评估跑分的完整流水线

## v3 重大升级

| 特性 | v2 | v3 |
|------|----|----|
| 分词器 | ❌ 假 BPE (260/32000 词) | ✅ 真正 BPE 训练 (32K 词) |
| 混合精度 | ❌ FP32 only | ✅ torch.amp autocast + GradScaler |
| 梯度累积 | ❌ 无 | ✅ 可配置 (默认8步, 有效batch=128) |
| 评估 | ❌ 无 | ✅ MMLU/C-Eval/GSM8K/HumanEval/HellaSwag |
| 检查点恢复 | ❌ 从零开始 | ✅ 保存优化器+调度器+步数 |
| Flash Attention | ❌ O(n²) | ✅ scaled_dot_product_attention |
| KV Cache 生成 | ❌ 无 | ✅ prefill + decode |
| 数据管线 | ❌ 单文件 | ✅ 多格式+数据混合+质量过滤+去重 |
| 分组量化 | ❌ 简单 INT4 | ✅ GPTQ 风格 128 元素分组 |
| 量化评估 | ❌ 空函数 | ✅ MSE + 余弦相似度 |

## 模型配置

| 模型 | 参数量 | d_model | n_heads | n_kv_heads | n_layers | d_ff | 量化大小 |
|------|--------|---------|---------|------------|----------|------|----------|
| FishAI-S | ~34M | 512 | 8 | 4 | 6 | 1408 | ~12MB |
| FishAI-M | ~400M | 896 | 14 | 2 | 24 | 4864 | ~150MB |
| FishAI-L | ~1.5B | 1536 | 12 | 4 | 28 | 8960 | ~500MB |

## 基准目标

| 基准 | FishAI-S (34M) | FishAI-M (400M) | FishAI-L (1.5B) | Qwen2.5-0.5B | Qwen2.5-1.5B |
|------|----------------|-----------------|-----------------|-------------|-------------|
| MMLU | ~5-15% | >40% | >55% | 47.5% | 60.9% |
| C-Eval | ~5-10% | >35% | >50% | 41.2% | 57.8% |
| GSM8K | ~0-5% | >25% | >55% | 49.6% | 73.2% |
| HumanEval | ~0-3% | >15% | >40% | 35.4% | 61.6% |

## 项目结构

```
fishai-train/
├── model.py            # 模型架构 (多尺寸配置, Flash Attention, KV Cache)
├── train.py            # 训练管线 (混合精度, 梯度累积, 评估, 恢复)
├── quantize.py         # 量化导出 (分组量化, 二进制格式, 量化评估)
├── evaluate.py         # 基准测试 (MMLU, C-Eval, GSM8K, HumanEval, HellaSwag)
├── benchmark.py        # 标准跑分框架 (对标 Pythia-70M / GPT-2 Small)
├── quick_train.py      # 一体化训练+评估脚本
├── reference_benchmark.py  # HuggingFace 参考模型基线
├── tokenizer_train.py  # BPE 分词器训练 (32K 词表, 特殊 token)
├── data_utils.py       # 数据处理 (下载, 过滤, 去重, 混合, 统计)
└── requirements.txt    # 依赖
```

## 跑分框架

### 快速自测 (验证架构正确性)
```bash
python benchmark.py self-test --config small
```
输出示例:
- 参数量: 34.1M
- 前向传播: ✅
- 生成功能: ✅
- KV Cache: ✅
- RoPE/SwiGLU/RMSNorm/GQA/WeightTying: ✅

### 完整跑分 (需要预训练模型)
```bash
python benchmark.py full --model checkpoints/fishai-best.pt --tokenizer tokenizer/
```

### 参考基线
```bash
python reference_benchmark.py
```

### 对标目标

| 模型 | 参数量 | WikiText-103 PPL | WikiText-2 PPL | HellaSwag |
|------|--------|-------------------|-----------------|-----------|
| Pythia-70M | 70M | ~56.0 | ~42.0 | ~26.3% |
| GPT-2 Small | 124M | ~37.5 | ~29.0 | ~31.0% |
| Pythia-160M | 160M | ~36.8 | ~27.0 | ~30.8% |
| SmolLM2-135M | 135M | ~42.0 | ~32.0 | ~31.5% |
| **FishAI-S (目标)** | 34M | ≤56.0 | ≤42.0 | - |

> FishAI-S 参数量仅 34M (比 Pythia-70M 少一半), 目标 PPL 追平 Pythia-70M

## 训练进展

- ✅ 架构验证: RoPE + SwiGLU + RMSNorm + GQA + WeightTying 全部实现
- ✅ 200步训练: 随机数据 PPL 从 ~32000 降到 61.7 (证明 loss 正常下降)
- ⏳ 真实数据训练: 需要在 GPU 上使用 WikiText-103/FineWeb-Edu 训练 50K+ 步
- ⏳ 跑分对比: 训练完成后运行 benchmark.py 对标 Pythia-70M

## 训练流程

```bash
# 1. 准备数据
python data_utils.py --output_dir data/ --mix web:50%,code:20%,books:15%,wiki:15%

# 2. 训练分词器
python tokenizer_train.py --data data/train.txt --vocab_size 32000 --output weights/tokenizer.json

# 3. 训练模型 (小模型)
python train.py --model_size small --data_path data/train.txt --tokenizer_path weights/tokenizer.json --max_steps 50000

# 4. 训练模型 (中等模型, 需要 GPU)
python train.py --model_size medium --data_path data/train.txt --tokenizer_path weights/tokenizer.json --max_steps 200000 --gradient_accumulation 8 --fp16

# 5. 量化导出
python quantize.py --checkpoint checkpoints/best_model.pt --output weights/model_q4.fq3 --format binary

# 6. 评估跑分
python evaluate.py --model_path checkpoints/best_model.pt --tokenizer_path weights/tokenizer.json --benchmarks mmlu,c_eval,gsm8k
```

## 关键训练策略

### 数据配比 (推荐)
- Web 文本: 50-60% (FineWeb-Edu, DCLM)
- 代码: 15-20% (The Stack, StarCoder)
- 书籍: 10-15% (Books3)
- 百科/维基: 10-15%
- 数学/科学: 5-10%

### Chinchilla 最优 vs 过度训练
- Chinchilla 最优: 34M 参数 → ~680M tokens
- Qwen2.5-0.5B 实际训练: 18T tokens (26,000× Chinchilla 最优!)
- 结论: **过度训练是关键** — 小模型需要远超 Chinchilla 最优的 token 量

### 知识蒸馏
- 从大模型 (Qwen2.5-7B) 蒸馏到 FishAI-M/L
- 使用 KL 散度损失 + 硬标签损失的加权组合
- 蒸馏可提升 MMLU 5-15%

## 路线图

- [ ] 训练 FishAI-M (400M) 到收敛
- [ ] C-Eval/MMLU 跑分 > 40
- [ ] 知识蒸馏实验
- [ ] DPO 对齐训练
- [ ] FishAI-L (1.5B) 训练

## 许可证

MIT License - FishLab-ai
