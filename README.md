# 🔬 FishAI Train v2

> FishLab-ai 自研 GPT 训练管线 — LLaMA-style 架构，从模型定义到量化导出

## v2 训练升级

- **RoPE** 旋转位置编码 (零参数位置编码)
- **SwiGLU** 激活函数 (三矩阵 FFN，更强表达力)
- **RMSNorm** (比 LayerNorm 更快更简)
- **GQA** 分组查询注意力 (KV 头数 < Q 头数)
- **权重绑定** (Token Embed 与 LM Head 共享)
- **无偏置** (现代发现 bias 在 RMSNorm+Residual 下冗余)
- **过度训练策略** (Chinchilla 的 20× → 100-500×)

## 训练策略

| 参数 | 值 |
|------|---|
| 优化器 | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| 学习率 | 6e-4 → 6e-5 (cosine decay) |
| Warmup | 1000 steps |
| 梯度裁剪 | 1.0 |
| 数据混合 | 50-60% web + 15-20% code + 10-15% books |

## 量化策略

| 层 | 精度 | 原因 |
|----|------|------|
| Token Embedding | FP16 | 精度敏感 |
| RMSNorm gamma | FP16 | 精度敏感 |
| Q/K 投影 | INT8 | 注意力精度敏感 |
| V/O 投影 | INT4 | 对量化鲁棒 |
| FFN (gate/up/down) | INT4 | 对量化鲁棒 |

## 项目结构

```
fishai-train/
├── model.py          # FishAI v2 模型 (RoPE+SwiGLU+RMSNorm+GQA+WeightTying)
├── train.py          # 训练主脚本 (AdamW + Cosine LR + 梯度裁剪)
├── quantize.py       # 混合精度量化导出 (FP16+INT8+INT4)
├── requirements.txt  # Python 依赖
└── data/             # 训练数据目录
```

## 训练

```bash
pip install -r requirements.txt
python train.py
```

## 导出量化权重

```python
from quantize import export_quantized_weights
from model import GPT, GPTConfig

model = GPT(GPTConfig())
export_quantized_weights(model, GPTConfig(), "model_q4.json")
```

## 许可证

MIT License - FishLab-ai
