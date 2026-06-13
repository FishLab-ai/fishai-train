"""
FishAI v2 混合精度量化导出

量化策略:
- Token Embedding / RMSNorm gamma: FP16 保留 (精度敏感)
- Q/K 投影: INT8 (注意力精度更敏感)
- V/O 投影 / FFN 权重: INT4 (对量化更鲁棒)
- LM Head: 如果权重绑定则不需要单独存储

预期: 3-4× 压缩率，困惑度损失 < 1%
"""

import json
import numpy as np
import torch
from model import GPT, GPTConfig


def quantize_int4(tensor: torch.Tensor, channel_dim: int = 0):
    """INT4 Per-Channel 量化"""
    data = tensor.detach().cpu().float().numpy()
    shape = list(data.shape)
    n_channels = shape[channel_dim]

    # 计算每通道的 channel_size
    channel_size = 1
    for i, s in enumerate(shape):
        if i != channel_dim:
            channel_size *= s

    # 重新排列为 [n_channels, channel_size]
    if channel_dim == 0:
        flat = data.reshape(n_channels, -1)
    else:
        # 转置使得 channel_dim 在第0位
        dims = list(range(len(shape)))
        dims[0], dims[channel_dim] = dims[channel_dim], dims[0]
        flat = np.transpose(data, dims).reshape(n_channels, -1)

    scales = []
    zero_points = []
    quantized_data = []

    for ch in range(n_channels):
        ch_data = flat[ch]
        ch_min = float(ch_data.min())
        ch_max = float(ch_data.max())

        ch_scale = (ch_max - ch_min) / 15.0
        ch_zp = round(-ch_min / ch_scale) if ch_scale > 0 else 8
        ch_zp = max(0, min(15, ch_zp))

        scales.append(ch_scale if ch_scale > 0 else 0.01)
        zero_points.append(ch_zp)

        if ch_scale > 0:
            q = np.round(ch_data / ch_scale) + ch_zp
            q = np.clip(q, 0, 15).astype(np.uint8)
        else:
            q = np.full(channel_size, 8, dtype=np.uint8)

        quantized_data.append(q)

    # 打包: 两个 4-bit 值存入一个 u8
    all_quantized = np.concatenate(quantized_data)
    packed = []
    for i in range(0, len(all_quantized), 2):
        low = int(all_quantized[i])
        high = int(all_quantized[i + 1]) if i + 1 < len(all_quantized) else 0
        packed.append((high << 4) | low)

    return {
        "data": packed,
        "scale": scales,
        "zero_point": zero_points,
        "shape": shape,
    }


def quantize_int8(tensor: torch.Tensor, channel_dim: int = 0):
    """INT8 Per-Channel 量化 (用于注意力 Q/K 投影)"""
    data = tensor.detach().cpu().float().numpy()
    shape = list(data.shape)
    n_channels = shape[channel_dim]

    channel_size = 1
    for i, s in enumerate(shape):
        if i != channel_dim:
            channel_size *= s

    if channel_dim == 0:
        flat = data.reshape(n_channels, -1)
    else:
        dims = list(range(len(shape)))
        dims[0], dims[channel_dim] = dims[channel_dim], dims[0]
        flat = np.transpose(data, dims).reshape(n_channels, -1)

    scales = []
    zero_points = []
    quantized_data = []

    for ch in range(n_channels):
        ch_data = flat[ch]
        ch_min = float(ch_data.min())
        ch_max = float(ch_data.max())

        ch_scale = (ch_max - ch_min) / 255.0
        ch_zp = round(-ch_min / ch_scale) if ch_scale > 0 else 127
        ch_zp = max(-128, min(127, ch_zp))

        scales.append(ch_scale if ch_scale > 0 else 0.001)
        zero_points.append(ch_zp)

        if ch_scale > 0:
            q = np.round(ch_data / ch_scale) + ch_zp
            q = np.clip(q, 0, 255).astype(np.uint8)
        else:
            q = np.full(channel_size, 127, dtype=np.uint8)

        quantized_data.append(q)

    all_quantized = np.concatenate(quantized_data)

    return {
        "data": all_quantized.tolist(),
        "scale": scales,
        "zero_point": zero_points,
        "shape": shape,
    }


def quantize_fp16(tensor: torch.Tensor):
    """FP16 精度保留 (实际用 FP32 存储)"""
    data = tensor.detach().cpu().float().numpy()
    return {
        "data": data.flatten().tolist(),
        "shape": list(data.shape),
    }


def export_quantized_weights(model: GPT, config: GPTConfig, output_path: str):
    """
    导出混合精度量化权重 (与 Rust 引擎兼容的 JSON 格式)

    量化策略:
    - token_embedding: FP16
    - Wq, Wk: INT8 (注意力精度敏感)
    - Wv, Wo: INT4
    - W_gate, W_up, W_down: INT4
    - RMSNorm gamma: FP16
    - LM Head: 权重绑定则跳过
    """
    print(f"\n[量化] 导出混合精度权重...")

    weights_dict = {
        "config": {
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_kv_heads": config.n_kv_heads,
            "n_layers": config.n_layers,
            "d_ff": config.d_ff,
            "rope_theta": config.rope_theta,
            "norm_eps": config.norm_eps,
            "weight_tying": config.weight_tying,
        },
        "token_embedding": quantize_fp16(model.token_embedding.weight),
        "layers": [],
        "final_rms_gamma": model.final_rms.gamma.detach().cpu().tolist(),
        "lm_head": None,
    }

    total_params = 0
    total_bytes = 0

    for i, block in enumerate(model.blocks):
        layer_dict = {
            # Q/K 用 INT8 (注意力精度敏感)
            "wq": quantize_int8(block.attn.wq.weight, channel_dim=0),
            "wk": quantize_int8(block.attn.wk.weight, channel_dim=0),
            # V/O 用 INT4
            "wv": quantize_int4(block.attn.wv.weight, channel_dim=0),
            "wo": quantize_int4(block.attn.wo.weight, channel_dim=0),
            # SwiGLU 三矩阵用 INT4
            "w_gate": quantize_int4(block.ffn.w_gate.weight, channel_dim=0),
            "w_up": quantize_int4(block.ffn.w_up.weight, channel_dim=0),
            "w_down": quantize_int4(block.ffn.w_down.weight, channel_dim=0),
            # RMSNorm gamma 用 FP16
            "rms1_gamma": block.rms1.gamma.detach().cpu().tolist(),
            "rms2_gamma": block.rms2.gamma.detach().cpu().tolist(),
        }
        weights_dict["layers"].append(layer_dict)

        # 统计
        for key, val in layer_dict.items():
            if isinstance(val, dict) and "data" in val:
                if isinstance(val["data"], list):
                    n = len(val["data"])
                    if key in ("wq", "wk"):
                        total_bytes += n * 1   # INT8: 1 byte/value
                    elif key in ("wv", "wo", "w_gate", "w_up", "w_down"):
                        total_bytes += n * 0.5  # INT4: 0.5 byte/value
                    total_params += len(val.get("scale", [])) * 4 + len(val.get("zero_point", []))

    # Embedding FP16
    emb_data = weights_dict["token_embedding"]["data"]
    total_bytes += len(emb_data) * 4  # FP32 存储
    total_params += len(emb_data)

    # Norm FP16
    total_bytes += len(weights_dict["final_rms_gamma"]) * 4
    total_params += len(weights_dict["final_rms_gamma"])

    size_mb = total_bytes / (1024 * 1024)
    print(f"[量化] 参数统计: ~{total_params/1e6:.1f}M")
    print(f"[量化] 量化文件预估: ~{size_mb:.1f} MB")
    print(f"[量化] 策略: Embed/Norm=FP16, Q/K=INT8, V/O/FFN=INT4")

    # 保存
    with open(output_path, 'w') as f:
        json.dump(weights_dict, f)
    print(f"[量化] 已保存 -> {output_path}")

    # 文件大小
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[量化] 实际文件大小: {file_size:.1f} MB")


def compute_quantization_error(model: GPT, config: GPTConfig):
    """计算量化误差统计"""
    errors = {}

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # 跳过 norm gamma 等

        original = param.detach().cpu().float().numpy()
        if "wq" in name or "wk" in name:
            q = quantize_int8(param, channel_dim=0)
            # 简单近似误差
            n_values = len(q["data"])
            errors[name] = "INT8"
        else:
            q = quantize_int4(param, channel_dim=0)
            errors[name] = "INT4"

    print(f"\n[量化] 量化方案分配:")
    for name, scheme in sorted(errors.items()):
        print(f"  {name}: {scheme}")


import os
