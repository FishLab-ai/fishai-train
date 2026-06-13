# 🔬 FishAI Train

> FishLab-ai 自研 GPT 训练管线 — 从模型定义到量化导出

## 概述

FishAI Train 是配套 FishAI Engine 的训练管线，完全自研。

## 项目结构

```
fishai-train/
├── model.py          # GPT 模型定义 (PyTorch, 从零自研)
├── train.py          # 训练主脚本
├── quantize.py       # 4-bit 量化导出
├── requirements.txt  # Python 依赖
└── data/             # 训练数据目录
```

## 训练

```bash
pip install -r requirements.txt
python train.py --data ./data --epochs 10
```

## 许可证

MIT License - FishLab-ai
