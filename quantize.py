"""
FishAI - 4-bit 整数量化导出

量化方案: INT4 Per-Channel 对称量化
- 量化公式: value = (int4 - zero_point) * scale
- Per-Channel: 每个输出通道独立的 scale 和 zero_point
- 存储: 每个 u8 存储 2 个 4-bit 值

量化流程:
1. 加载 FP32 模型权重
2. 对每个权重矩阵按通道量化
3. 打包为紧凑格式
4. 导出为 JSON (兼容 Rust 引擎读取)
"""

import json
import struct
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any

from model import GPT, GPTConfig


def quantize_tensor_to_int4(
    weight: np.ndarray,
    channel_dim: int = 0,
) -> Dict[str, Any]:
    """
    将 FP32 权重量化为 INT4

    Args:
        weight: numpy 数组
        channel_dim: 通道维度

    Returns:
        包含量化数据的字典:
        - data: u8 数组 (每字节存2个4-bit值)
        - scale: 每通道的缩放因子
        - zero_point: 每通道的零点
        - shape: 原始形状
    """
    shape = list(weight.shape)
    n_channels = shape[channel_dim]

    # 计算每个通道的 channel_size
    channel_size = 1
    for i, s in enumerate(shape):
        if i != channel_dim:
            channel_size *= s

    # 重塑为 [n_channels, channel_size]
    weight_2d = np.moveaxis(weight, channel_dim, 0).reshape(n_channels, -1)

    scale = np.zeros(n_channels, dtype=np.float32)
    zero_point = np.zeros(n_channels, dtype=np.int8)
    quantized = np.zeros_like(weight_2d, dtype=np.uint8)

    for ch in range(n_channels):
        ch_data = weight_2d[ch]
        ch_min = ch_data.min()
        ch_max = ch_data.max()

        # 计算 scale 和 zero_point
        ch_scale = (ch_max - ch_min) / 15.0
        if ch_scale == 0:
            ch_scale = 1e-8

        ch_zp = int(round(-ch_min / ch_scale))
        ch_zp = max(0, min(15, ch_zp))

        scale[ch] = ch_scale
        zero_point[ch] = ch_zp

        # 量化
        q = np.round(ch_data / ch_scale) + ch_zp
        q = np.clip(q, 0, 15).astype(np.uint8)
        quantized[ch] = q

    # 重塑回原始形状
    quantized = quantized.reshape([n_channels] + [s for i, s in enumerate(shape) if i != channel_dim])
    quantized = np.moveaxis(quantized, 0, channel_dim)
    quantized = quantized.reshape(shape)

    # 打包: 2 个 4-bit 值 -> 1 个 u8
    flat = quantized.flatten()
    # 偶数位放低4位, 奇数位放高4位
    packed_len = (len(flat) + 1) // 2
    packed = np.zeros(packed_len, dtype=np.uint8)

    for i in range(0, len(flat), 2):
        low = flat[i] & 0x0F
        if i + 1 < len(flat):
            high = flat[i + 1] & 0x0F
        else:
            high = 0
        packed[i // 2] = low | (high << 4)

    return {
        "data": packed.tolist(),
        "scale": scale.tolist(),
        "zero_point": zero_point.tolist(),
        "shape": shape,
    }


def export_quantized_weights(
    model: GPT,
    config: GPTConfig,
    output_path: Path,
):
    """
    导出整个模型的 4-bit 量化权重

    输出格式兼容 Rust 引擎的 GPTWeights 结构
    """
    weights = {
        "config": {
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
        },
        "token_embedding": quantize_tensor_to_int4(
            model.token_embedding.weight.detach().numpy()
        ),
        "position_embedding": quantize_tensor_to_int4(
            model.position_embedding.weight.detach().numpy()
        ),
        "layers": [],
        "final_ln_gamma": model.final_ln.weight.detach().numpy().tolist(),
        "final_ln_beta": model.final_ln.bias.detach().numpy().tolist(),
        "lm_head": quantize_tensor_to_int4(
            model.lm_head.weight.detach().numpy()
        ),
    }

    # 每层 Transformer
    for i, block in enumerate(model.blocks):
        layer_weights = {
            "wq": quantize_tensor_to_int4(
                block.attn.qkv_proj.weight[:config.d_model].detach().numpy()
            ),
            "wk": quantize_tensor_to_int4(
                block.attn.qkv_proj.weight[config.d_model:2*config.d_model].detach().numpy()
            ),
            "wv": quantize_tensor_to_int4(
                block.attn.qkv_proj.weight[2*config.d_model:].detach().numpy()
            ),
            "wo": quantize_tensor_to_int4(
                block.attn.out_proj.weight.detach().numpy()
            ),
            "w1": quantize_tensor_to_int4(
                block.ffn.net[0].weight.detach().numpy()
            ),
            "w2": quantize_tensor_to_int4(
                block.ffn.net[2].weight.detach().numpy()
            ),
            "ln1_gamma": block.ln1.weight.detach().numpy().tolist(),
            "ln1_beta": block.ln1.bias.detach().numpy().tolist(),
            "ln2_gamma": block.ln2.weight.detach().numpy().tolist(),
            "ln2_beta": block.ln2.bias.detach().numpy().tolist(),
        }
        weights["layers"].append(layer_weights)
        print(f"  Layer {i+1}/{config.n_layers} 量化完成")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(weights, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[导出] 量化权重大小: {size_mb:.1f} MB")
    print(f"[导出] 保存到: {output_path}")

    return weights


def compute_quantization_error(
    original: np.ndarray,
    quantized_dict: Dict[str, Any],
) -> float:
    """
    计算量化误差 (MSE)
    """
    # 解量化
    shape = quantized_dict["shape"]
    packed = np.array(quantized_dict["data"], dtype=np.uint8)
    scale = np.array(quantized_dict["scale"], dtype=np.float32)
    zero_point = np.array(quantized_dict["zero_point"], dtype=np.int8)

    # 解包
    flat = np.zeros(np.prod(shape), dtype=np.uint8)
    for i in range(len(packed)):
        flat[2*i] = packed[i] & 0x0F
        if 2*i + 1 < len(flat):
            flat[2*i + 1] = (packed[i] >> 4) & 0x0F

    # 解量化 (简化版)
    n_channels = len(scale)
    channel_size = len(flat) // n_channels
    dequantized = np.zeros_like(flat, dtype=np.float32)
    for ch in range(n_channels):
        start = ch * channel_size
        end = start + channel_size
        dequantized[start:end] = (flat[start:end].astype(np.float32) - zero_point[ch]) * scale[ch]

    # MSE
    mse = np.mean((original.flatten()[:len(dequantized)] - dequantized) ** 2)
    return float(mse)


if __name__ == "__main__":
    # 测试量化
    print("测试 INT4 量化...")

    config = GPTConfig()
    model = GPT(config)

    # 测试单个张量量化
    test_weight = np.random.randn(512, 512).astype(np.float32) * 0.02
    q_dict = quantize_tensor_to_int4(test_weight)
    mse = compute_quantization_error(test_weight, q_dict)
    print(f"单层量化 MSE: {mse:.6f}")

    # 测试完整模型导出
    output_path = Path("./test_quantized.json")
    export_quantized_weights(model, config, output_path)

    # 清理
    output_path.unlink(missing_ok=True)
    print("\n量化测试通过!")
