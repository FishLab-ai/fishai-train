# 🔬 TinyAI Train

> 完全自研的 GPT 训练管线 — 从模型定义到量化导出

## 概述

TinyAI Train 是配套 TinyAI Engine 的训练管线，包含完整的模型定义、训练脚本、分词器训练和 4-bit 量化导出工具。

### 训练流程

```
语料数据 → 分词器训练 → 模型训练 → 4-bit 量化 → 权重导出
   ↓           ↓           ↓          ↓          ↓
 data/    tokenizer.json  best.pt   quantize   model_q4.json
```

## 项目结构

```
tinyai-train/
├── model.py          # GPT 模型定义 (PyTorch)
├── train.py          # 训练主脚本
├── quantize.py       # 4-bit 量化导出
├── requirements.txt  # Python 依赖
└── data/             # 训练数据目录
```

## 自研模型架构 (model.py)

从零实现每一个组件：

1. **CausalSelfAttention** — 多头因果自注意力
   - Q, K, V 合并投影 (效率优化)
   - Scaled Dot-Product Attention
   - 因果掩码 (下三角)
   - Attention Dropout + Residual Dropout

2. **FeedForward** — 前馈网络
   - `Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model)`
   - Dropout 正则化

3. **TransformerBlock** — Pre-LayerNorm Transformer Block
   - Pre-Norm: 先归一化再计算 (比 Post-Norm 更稳定)
   - 残差连接

4. **GPT** — 完整模型
   - Token Embedding + Position Embedding (可学习)
   - N 层 Transformer Block
   - 最终 LayerNorm + LM Head
   - 权重初始化: Normal(0, 0.02)
   - 支持 Top-K 和 Nucleus (Top-P) 采样

## 训练参数

```bash
python train.py \
  --data ./data \
  --output-dir ./output \
  --vocab-size 32000 \
  --d-model 512 \
  --n-heads 8 \
  --n-layers 6 \
  --d-ff 2048 \
  --max-seq-len 512 \
  --epochs 10 \
  --batch-size 8 \
  --lr 3e-4 \
  --weight-decay 0.1
```

## 4-bit 量化导出 (quantize.py)

### 量化方案
- **INT4 Per-Channel 对称量化**
- 量化公式: `value = (int4 - zero_point) * scale`
- 紧凑存储: 每个 u8 存储 2 个 4-bit 值

### 量化流程
1. 加载 FP32 模型权重
2. 对每个权重矩阵按输出通道量化
3. 计算 per-channel 的 scale 和 zero_point
4. 打包为紧凑格式 (2 个 4-bit → 1 个 u8)
5. 导出为 JSON (兼容 Rust 引擎读取)

## 安装 & 运行

```bash
pip install -r requirements.txt

# 准备训练数据 (放到 data/ 目录，支持 .txt 和 .jsonl)
# .jsonl 格式: {"text": "你的训练文本"}

# 开始训练
python train.py --data ./data --epochs 10

# 训练完成后，量化权重会自动导出到 output/model_q4.json
# 将 output/model_q4.json 和 output/tokenizer.json 复制到 tinyai-engine/weights/
```

## 推荐训练数据

- 中文维基百科
- 英文维基百科
- GitHub 代码数据集 (The Stack)
- 自定义领域语料

## 许可证

MIT License - FishLab-ai
