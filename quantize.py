"""
FishAI v3 混合精度量化导出

v3 量化升级:
- 修复: compute_quantization_error() 真正计算 MSE 和余弦相似度
- 修复: quantize_fp16() 存储真实 FP16 (torch.float16)
- 新增: 分组量化 (128 元素一组, 类 GPTQ)
- 新增: 二进制导出格式 (.bin) 替代 JSON
- 新增: 量化感知评估: 运行推理步骤比较输出

量化策略:
- Token Embedding / RMSNorm gamma: FP16 保留 (精度敏感)
- Q/K 投影: INT8 (注意力精度更敏感)
- V/O 投影 / FFN 权重: INT4 (对量化更鲁棒)
- 分组量化: 128 元素一组, 每组独立 scale/zero_point

预期: 3-4× 压缩率，困惑度损失 < 1%
"""

import os
import json
import struct
import argparse
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPT, GPTConfig


# ──────────────── 量化常量 ────────────────

QUANT_MAGIC = b"FAQM"   # FishAI Quantized Model
QUANT_VERSION = 2

# 量化类型标识
QUANT_FP16 = 0
QUANT_INT8 = 1
QUANT_INT4 = 2
QUANT_INT4_GROUP128 = 3  # 分组量化, 128 元素一组


# ──────────────── INT4 分组量化 ────────────────

def quantize_int4_group128(
    tensor: torch.Tensor,
    group_size: int = 128,
) -> Dict[str, Any]:
    """
    INT4 分组量化 (类 GPTQ)

    将权重按 group_size 分组，每组独立计算 scale 和 zero_point
    比逐通道量化更精细，减少量化误差

    Args:
        tensor: 待量化的权重张量
        group_size: 分组大小 (默认 128)

    Returns:
        {
            "data": packed_4bit (numpy uint8 array, 2个4bit打包为1个uint8),
            "scales": 每组的 scale (float16 numpy array),
            "zero_points": 每组的 zero_point (uint8 numpy array),
            "shape": 原始形状,
            "group_size": 分组大小,
        }
    """
    data = tensor.detach().cpu().float().numpy()
    shape = data.shape
    flat = data.flatten()
    n_elements = flat.size

    # 填充到 group_size 的倍数
    pad_size = (group_size - n_elements % group_size) % group_size
    if pad_size > 0:
        flat = np.concatenate([flat, np.zeros(pad_size, dtype=np.float32)])

    n_groups = flat.size // group_size
    grouped = flat.reshape(n_groups, group_size)

    # 逐组量化
    scales = np.zeros(n_groups, dtype=np.float16)
    zero_points = np.zeros(n_groups, dtype=np.uint8)
    quantized = np.zeros(n_groups * group_size, dtype=np.uint8)

    for g in range(n_groups):
        group_data = grouped[g]
        g_min = float(group_data.min())
        g_max = float(group_data.max())

        # INT4: 值域 [0, 15]
        scale = (g_max - g_min) / 15.0
        zp = round(-g_min / scale) if scale > 0 else 8
        zp = max(0, min(15, zp))

        scales[g] = scale if scale > 0 else np.float16(0.01)
        zero_points[g] = zp

        if scale > 0:
            q = np.round(group_data / scale) + zp
            q = np.clip(q, 0, 15).astype(np.uint8)
        else:
            q = np.full(group_size, 8, dtype=np.uint8)

        quantized[g * group_size:(g + 1) * group_size] = q

    # 去除填充
    quantized = quantized[:n_elements]

    # 打包: 两个 4-bit 值存入一个 uint8
    packed = np.zeros((n_elements + 1) // 2, dtype=np.uint8)
    for i in range(0, n_elements, 2):
        low = int(quantized[i])
        high = int(quantized[i + 1]) if i + 1 < n_elements else 0
        packed[i // 2] = (high << 4) | low

    return {
        "data": packed,
        "scales": scales,
        "zero_points": zero_points,
        "shape": list(shape),
        "group_size": group_size,
        "quant_type": QUANT_INT4_GROUP128,
    }


def dequantize_int4_group128(q_data: Dict[str, Any]) -> np.ndarray:
    """
    INT4 分组量化的反量化

    Args:
        q_data: quantize_int4_group128 的输出

    Returns:
        反量化后的 float32 numpy 数组
    """
    packed = q_data["data"]
    scales = q_data["scales"].astype(np.float32)
    zero_points = q_data["zero_points"]
    shape = q_data["shape"]
    group_size = q_data["group_size"]

    # 解包
    n_elements = int(np.prod(shape))
    quantized = np.zeros(n_elements, dtype=np.uint8)
    for i in range(n_elements):
        byte_idx = i // 2
        if i % 2 == 0:
            quantized[i] = packed[byte_idx] & 0x0F
        else:
            quantized[i] = (packed[byte_idx] >> 4) & 0x0F

    # 逐组反量化
    flat = np.zeros(n_elements, dtype=np.float32)
    pad_size = (group_size - n_elements % group_size) % group_size
    if pad_size > 0:
        quantized = np.concatenate([quantized, np.zeros(pad_size, dtype=np.uint8)])

    n_groups = quantized.size // group_size
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group_q = quantized[start:end].astype(np.float32)
        group_deq = (group_q - zero_points[g]) * scales[g]
        # 只写入实际元素
        actual_end = min(end, n_elements)
        flat[start:actual_end] = group_deq[:actual_end - start]

    return flat.reshape(shape)


# ──────────────── INT4 逐通道量化 ────────────────

def quantize_int4(tensor: torch.Tensor, channel_dim: int = 0) -> Dict[str, Any]:
    """
    INT4 Per-Channel 量化

    Args:
        tensor: 待量化的权重张量
        channel_dim: 通道维度

    Returns:
        量化数据字典
    """
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

    all_quantized = np.concatenate(quantized_data)

    # 打包
    packed = []
    for i in range(0, len(all_quantized), 2):
        low = int(all_quantized[i])
        high = int(all_quantized[i + 1]) if i + 1 < len(all_quantized) else 0
        packed.append((high << 4) | low)

    return {
        "data": np.array(packed, dtype=np.uint8),
        "scale": np.array(scales, dtype=np.float16),
        "zero_point": np.array(zero_points, dtype=np.uint8),
        "shape": shape,
        "quant_type": QUANT_INT4,
    }


def dequantize_int4(q_data: Dict[str, Any]) -> np.ndarray:
    """INT4 Per-Channel 反量化"""
    packed = q_data["data"]
    scales = q_data["scale"].astype(np.float32)
    zero_points = q_data["zero_point"]
    shape = q_data["shape"]
    n_channels = shape[0]
    channel_size = int(np.prod(shape) // n_channels)

    # 解包
    n_elements = int(np.prod(shape))
    quantized = np.zeros(n_elements, dtype=np.uint8)
    for i in range(n_elements):
        byte_idx = i // 2
        if i % 2 == 0:
            quantized[i] = packed[byte_idx] & 0x0F
        else:
            quantized[i] = (packed[byte_idx] >> 4) & 0x0F

    # 逐通道反量化
    flat = quantized[:n_elements].reshape(n_channels, -1).astype(np.float32)
    for ch in range(n_channels):
        flat[ch] = (flat[ch] - zero_points[ch]) * scales[ch]

    return flat.reshape(shape)


# ──────────────── INT8 逐通道量化 ────────────────

def quantize_int8(tensor: torch.Tensor, channel_dim: int = 0) -> Dict[str, Any]:
    """
    INT8 Per-Channel 量化 (用于注意力 Q/K 投影)

    Args:
        tensor: 待量化的权重张量
        channel_dim: 通道维度

    Returns:
        量化数据字典
    """
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
        zero_points.append(int(ch_zp))

        if ch_scale > 0:
            q = np.round(ch_data / ch_scale) + ch_zp
            q = np.clip(q, 0, 255).astype(np.uint8)
        else:
            q = np.full(channel_size, 127, dtype=np.uint8)

        quantized_data.append(q)

    all_quantized = np.concatenate(quantized_data)

    return {
        "data": all_quantized,
        "scale": np.array(scales, dtype=np.float16),
        "zero_point": np.array(zero_points, dtype=np.int16),
        "shape": shape,
        "quant_type": QUANT_INT8,
    }


def dequantize_int8(q_data: Dict[str, Any]) -> np.ndarray:
    """INT8 Per-Channel 反量化"""
    data = q_data["data"].astype(np.float32)
    scales = q_data["scale"].astype(np.float32)
    zero_points = q_data["zero_point"].astype(np.float32)
    shape = q_data["shape"]
    n_channels = shape[0]
    channel_size = int(np.prod(shape) // n_channels)

    flat = data.reshape(n_channels, -1)
    for ch in range(n_channels):
        flat[ch] = (flat[ch] - zero_points[ch]) * scales[ch]

    return flat.reshape(shape)


# ──────────────── FP16 量化 ────────────────

def quantize_fp16(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    FP16 精度保留 — 真正存储为 FP16

    Args:
        tensor: 待量化的权重张量

    Returns:
        {"data": float16 numpy array, "shape": list}
    """
    # 关键修复: 真正转为 float16，不再是 float32
    data = tensor.detach().cpu().to(torch.float16).numpy()
    return {
        "data": data,
        "shape": list(data.shape),
        "quant_type": QUANT_FP16,
    }


def dequantize_fp16(q_data: Dict[str, Any]) -> np.ndarray:
    """FP16 反量化 (转为 float32)"""
    return q_data["data"].astype(np.float32).reshape(q_data["shape"])


# ──────────────── 量化误差计算 (修复!) ────────────────

def compute_quantization_error(
    model: GPT,
    config: GPTConfig,
    use_group_quant: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    计算量化误差统计 — 真正计算 MSE 和余弦相似度

    对每个权重矩阵:
    1. 量化 (INT4/INT8)
    2. 反量化
    3. 计算 MSE (均方误差)
    4. 计算余弦相似度
    5. 计算信噪比 (SNR)

    Args:
        model: 模型
        config: 模型配置
        use_group_quant: 是否使用分组量化

    Returns:
        {参数名: {"mse": ..., "cosine_sim": ..., "snr_db": ..., "quant_type": ...}}
    """
    errors = {}

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # 跳过 norm gamma 等 1D 参数

        original = param.detach().cpu().float().numpy()
        original_flat = original.flatten()

        # 选择量化方式
        if "wq" in name or "wk" in name:
            q_data = quantize_int8(param, channel_dim=0)
            dequantized = dequantize_int8(q_data)
            quant_type = "INT8"
        else:
            if use_group_quant:
                q_data = quantize_int4_group128(param, group_size=128)
                dequantized = dequantize_int4_group128(q_data)
                quant_type = "INT4-G128"
            else:
                q_data = quantize_int4(param, channel_dim=0)
                dequantized = dequantize_int4(q_data)
                quant_type = "INT4"

        # 确保形状一致
        dequantized = dequantized.reshape(original.shape)
        dequantized_flat = dequantized.flatten()

        # MSE (均方误差)
        mse = float(np.mean((original_flat - dequantized_flat) ** 2))

        # 余弦相似度
        norm_orig = np.linalg.norm(original_flat)
        norm_deq = np.linalg.norm(dequantized_flat)
        if norm_orig > 1e-8 and norm_deq > 1e-8:
            cosine_sim = float(
                np.dot(original_flat, dequantized_flat) / (norm_orig * norm_deq)
            )
        else:
            cosine_sim = 1.0

        # SNR (信噪比, dB)
        signal_power = float(np.mean(original_flat ** 2))
        noise_power = mse
        if noise_power > 1e-12:
            snr_db = 10.0 * np.log10(signal_power / noise_power)
        else:
            snr_db = float("inf")

        # 最大绝对误差
        max_abs_error = float(np.max(np.abs(original_flat - dequantized_flat)))

        errors[name] = {
            "mse": mse,
            "cosine_sim": cosine_sim,
            "snr_db": snr_db,
            "max_abs_error": max_abs_error,
            "quant_type": quant_type,
        }

    # 打印摘要
    print(f"\n{'='*80}")
    print(f"  量化误差统计 (分组量化={'是' if use_group_quant else '否'})")
    print(f"{'='*80}")
    print(f"  {'参数名':<40} {'量化':<10} {'MSE':<12} {'余弦相似度':<12} {'SNR(dB)':<10}")
    print(f"  {'-'*40} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    total_mse = 0.0
    total_cosine = 0.0
    n = 0

    for name, err in sorted(errors.items()):
        print(f"  {name:<40} {err['quant_type']:<10} {err['mse']:<12.6f} "
              f"{err['cosine_sim']:<12.6f} {err['snr_db']:<10.2f}")
        total_mse += err["mse"]
        total_cosine += err["cosine_sim"]
        n += 1

    if n > 0:
        print(f"\n  平均 MSE: {total_mse / n:.6f}")
        print(f"  平均余弦相似度: {total_cosine / n:.6f}")

    return errors


# ──────────────── 量化感知评估 ────────────────

@torch.no_grad()
def quantization_aware_eval(
    model: GPT,
    config: GPTConfig,
    test_input_ids: Optional[torch.Tensor] = None,
    n_steps: int = 5,
    use_group_quant: bool = True,
) -> Dict[str, Any]:
    """
    量化感知评估: 运行推理步骤比较原始模型和量化模型的输出差异

    Args:
        model: 模型
        config: 模型配置
        test_input_ids: 测试输入 (None 则随机生成)
        n_steps: 测试步数
        use_group_quant: 是否使用分组量化

    Returns:
        评估结果字典
    """
    device = next(model.parameters()).device
    model.eval()

    if test_input_ids is None:
        test_input_ids = torch.randint(0, config.vocab_size, (1, 64), device=device)

    # 原始模型输出
    original_logits, original_loss, _ = model(test_input_ids, labels=test_input_ids)

    # 量化-反量化所有参数
    quantized_state = {}
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue
        if "wq" in name or "wk" in name:
            q_data = quantize_int8(param, channel_dim=0)
            deq = dequantize_int8(q_data)
        else:
            if use_group_quant:
                q_data = quantize_int4_group128(param, group_size=128)
                deq = dequantize_int4_group128(q_data)
            else:
                q_data = quantize_int4(param, channel_dim=0)
                deq = dequantize_int4(q_data)

        quantized_state[name] = torch.tensor(
            deq.reshape(param.shape),
            dtype=param.dtype,
            device=param.device,
        )

    # 临时替换参数
    original_state = {}
    for name, param in model.named_parameters():
        if name in quantized_state:
            original_state[name] = param.data.clone()
            param.data = quantized_state[name]

    # 量化模型输出
    quant_logits, quant_loss, _ = model(test_input_ids, labels=test_input_ids)

    # 恢复原始参数
    for name, param in model.named_parameters():
        if name in original_state:
            param.data = original_state[name]

    # 计算差异
    logits_diff = (original_logits - quant_logits).float()
    logits_mse = float(torch.mean(logits_diff ** 2))
    logits_cosine = float(F.cosine_similarity(
        original_logits.float().flatten().unsqueeze(0),
        quant_logits.float().flatten().unsqueeze(0),
    ).item())

    loss_diff = abs(original_loss.item() - quant_loss.item()) if original_loss is not None and quant_loss is not None else 0.0

    results = {
        "logits_mse": logits_mse,
        "logits_cosine_similarity": logits_cosine,
        "loss_original": original_loss.item() if original_loss is not None else None,
        "loss_quantized": quant_loss.item() if quant_loss is not None else None,
        "loss_diff": loss_diff,
        "n_steps": n_steps,
        "use_group_quant": use_group_quant,
    }

    print(f"\n{'='*60}")
    print(f"  量化感知评估")
    print(f"{'='*60}")
    print(f"  原始 Loss:     {results['loss_original']:.4f}")
    print(f"  量化 Loss:     {results['loss_quantized']:.4f}")
    print(f"  Loss 差异:     {results['loss_diff']:.4f}")
    print(f"  Logits MSE:    {results['logits_mse']:.6f}")
    print(f"  Logits 余弦:   {results['logits_cosine_similarity']:.6f}")

    model.train()
    return results


# ──────────────── 二进制导出格式 ────────────────

def export_quantized_weights_binary(
    model: GPT,
    config: GPTConfig,
    output_path: str,
    use_group_quant: bool = True,
) -> None:
    """
    导出混合精度量化权重 (二进制格式)

    格式:
        头部:
            magic: 4 bytes = "FAQM"
            version: 4 bytes = uint32
            n_layers: 4 bytes = uint32
            config_json_len: 4 bytes = uint32
            config_json: variable bytes (UTF-8)
        权重数据:
            每个权重:
                name_len: 4 bytes = uint32
                name: variable bytes (UTF-8)
                quant_type: 4 bytes = uint32
                shape_len: 4 bytes = uint32
                shape: shape_len * 4 bytes (uint32 each)
                data_len: 4 bytes = uint32
                data: variable bytes (取决于量化类型)

    量化策略:
    - token_embedding: FP16
    - Wq, Wk: INT8 (注意力精度敏感)
    - Wv, Wo, W_gate, W_up, W_down: INT4 (分组量化)
    - RMSNorm gamma: FP16
    - LM Head: 权重绑定则跳过
    """
    print(f"\n[量化] 导出混合精度权重 (二进制格式)...")

    with open(output_path, "wb") as f:
        # ── 头部 ──
        f.write(QUANT_MAGIC)
        f.write(struct.pack("<I", QUANT_VERSION))
        f.write(struct.pack("<I", config.n_layers))

        # 配置 JSON
        config_json = json.dumps(config.to_dict(), ensure_ascii=False).encode("utf-8")
        f.write(struct.pack("<I", len(config_json)))
        f.write(config_json)

        # ── 写入 token embedding (FP16) ──
        emb_data = model.token_embedding.weight.detach().cpu().to(torch.float16).numpy()
        name = "token_embedding.weight"
        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", QUANT_FP16))
        f.write(struct.pack("<I", len(emb_data.shape)))
        for s in emb_data.shape:
            f.write(struct.pack("<I", s))
        emb_bytes = emb_data.tobytes()
        f.write(struct.pack("<I", len(emb_bytes)))
        f.write(emb_bytes)

        # ── 写入每层权重 ──
        for i, block in enumerate(model.blocks):
            layer_weights = {
                f"blocks.{i}.attn.wq.weight": (block.attn.wq.weight, "int8"),
                f"blocks.{i}.attn.wk.weight": (block.attn.wk.weight, "int8"),
                f"blocks.{i}.attn.wv.weight": (block.attn.wv.weight, "int4"),
                f"blocks.{i}.attn.wo.weight": (block.attn.wo.weight, "int4"),
                f"blocks.{i}.ffn.w_gate.weight": (block.ffn.w_gate.weight, "int4"),
                f"blocks.{i}.ffn.w_up.weight": (block.ffn.w_up.weight, "int4"),
                f"blocks.{i}.ffn.w_down.weight": (block.ffn.w_down.weight, "int4"),
                f"blocks.{i}.rms1.gamma": (block.rms1.gamma, "fp16"),
                f"blocks.{i}.rms2.gamma": (block.rms2.gamma, "fp16"),
            }

            for name, (weight, qtype) in layer_weights.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)

                if qtype == "fp16":
                    f.write(struct.pack("<I", QUANT_FP16))
                    data = weight.detach().cpu().to(torch.float16).numpy()
                    f.write(struct.pack("<I", len(data.shape)))
                    for s in data.shape:
                        f.write(struct.pack("<I", s))
                    data_bytes = data.tobytes()
                    f.write(struct.pack("<I", len(data_bytes)))
                    f.write(data_bytes)

                elif qtype == "int8":
                    f.write(struct.pack("<I", QUANT_INT8))
                    q_data = quantize_int8(weight, channel_dim=0)
                    f.write(struct.pack("<I", len(q_data["shape"])))
                    for s in q_data["shape"]:
                        f.write(struct.pack("<I", s))
                    # scales
                    scales_bytes = q_data["scale"].tobytes()
                    f.write(struct.pack("<I", len(scales_bytes)))
                    f.write(scales_bytes)
                    # zero_points
                    zp_bytes = q_data["zero_point"].tobytes()
                    f.write(struct.pack("<I", len(zp_bytes)))
                    f.write(zp_bytes)
                    # quantized data
                    data_bytes = q_data["data"].tobytes()
                    f.write(struct.pack("<I", len(data_bytes)))
                    f.write(data_bytes)

                elif qtype == "int4":
                    if use_group_quant:
                        f.write(struct.pack("<I", QUANT_INT4_GROUP128))
                        q_data = quantize_int4_group128(weight, group_size=128)
                        f.write(struct.pack("<I", len(q_data["shape"])))
                        for s in q_data["shape"]:
                            f.write(struct.pack("<I", s))
                        # group_size
                        f.write(struct.pack("<I", q_data["group_size"]))
                        # scales
                        scales_bytes = q_data["scales"].tobytes()
                        f.write(struct.pack("<I", len(scales_bytes)))
                        f.write(scales_bytes)
                        # zero_points
                        zp_bytes = q_data["zero_points"].tobytes()
                        f.write(struct.pack("<I", len(zp_bytes)))
                        f.write(zp_bytes)
                        # packed data
                        data_bytes = q_data["data"].tobytes()
                        f.write(struct.pack("<I", len(data_bytes)))
                        f.write(data_bytes)
                    else:
                        f.write(struct.pack("<I", QUANT_INT4))
                        q_data = quantize_int4(weight, channel_dim=0)
                        f.write(struct.pack("<I", len(q_data["shape"])))
                        for s in q_data["shape"]:
                            f.write(struct.pack("<I", s))
                        # scales
                        scales_bytes = q_data["scale"].tobytes()
                        f.write(struct.pack("<I", len(scales_bytes)))
                        f.write(scales_bytes)
                        # zero_points
                        zp_bytes = q_data["zero_point"].tobytes()
                        f.write(struct.pack("<I", len(zp_bytes)))
                        f.write(zp_bytes)
                        # packed data
                        data_bytes = q_data["data"].tobytes()
                        f.write(struct.pack("<I", len(data_bytes)))
                        f.write(data_bytes)

        # ── Final RMSNorm ──
        name = "final_rms.gamma"
        name_bytes = name.encode("utf-8")
        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", QUANT_FP16))
        data = model.final_rms.gamma.detach().cpu().to(torch.float16).numpy()
        f.write(struct.pack("<I", len(data.shape)))
        for s in data.shape:
            f.write(struct.pack("<I", s))
        data_bytes = data.tobytes()
        f.write(struct.pack("<I", len(data_bytes)))
        f.write(data_bytes)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[量化] 二进制导出完成: {file_size:.1f} MB -> {output_path}")


# ──────────────── JSON 导出 (兼容旧格式) ────────────────

def export_quantized_weights(
    model: GPT,
    config: GPTConfig,
    output_path: str,
    use_group_quant: bool = True,
) -> None:
    """
    导出混合精度量化权重 (JSON 格式, 兼容旧引擎)

    量化策略:
    - token_embedding: FP16 (真正存储为 float16)
    - Wq, Wk: INT8 (注意力精度敏感)
    - Wv, Wo: INT4
    - W_gate, W_up, W_down: INT4 (分组量化)
    - RMSNorm gamma: FP16
    - LM Head: 权重绑定则跳过
    """
    print(f"\n[量化] 导出混合精度权重 (JSON格式)...")

    weights_dict = {
        "config": config.to_dict(),
        "token_embedding": quantize_fp16(model.token_embedding.weight),
        "layers": [],
        "final_rms_gamma": model.final_rms.gamma.detach().cpu().to(torch.float16).tolist(),
        "lm_head": None,
    }

    total_bytes = 0

    for i, block in enumerate(model.blocks):
        layer_dict = {
            # Q/K 用 INT8
            "wq": quantize_int8(block.attn.wq.weight, channel_dim=0),
            "wk": quantize_int8(block.attn.wk.weight, channel_dim=0),
            # V/O 用 INT4 (分组量化)
            "wv": quantize_int4_group128(block.attn.wv.weight, group_size=128) if use_group_quant else quantize_int4(block.attn.wv.weight, channel_dim=0),
            "wo": quantize_int4_group128(block.attn.wo.weight, group_size=128) if use_group_quant else quantize_int4(block.attn.wo.weight, channel_dim=0),
            # SwiGLU 三矩阵用 INT4 (分组量化)
            "w_gate": quantize_int4_group128(block.ffn.w_gate.weight, group_size=128) if use_group_quant else quantize_int4(block.ffn.w_gate.weight, channel_dim=0),
            "w_up": quantize_int4_group128(block.ffn.w_up.weight, group_size=128) if use_group_quant else quantize_int4(block.ffn.w_up.weight, channel_dim=0),
            "w_down": quantize_int4_group128(block.ffn.w_down.weight, group_size=128) if use_group_quant else quantize_int4(block.ffn.w_down.weight, channel_dim=0),
            # RMSNorm gamma 用 FP16
            "rms1_gamma": block.rms1.gamma.detach().cpu().to(torch.float16).tolist(),
            "rms2_gamma": block.rms2.gamma.detach().cpu().to(torch.float16).tolist(),
        }
        weights_dict["layers"].append(layer_dict)

    # JSON 序列化需要转换 numpy 数组
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        raise TypeError(f"不可序列化的类型: {type(obj)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f, default=convert_numpy)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[量化] JSON 导出完成: {file_size:.1f} MB -> {output_path}")
    print(f"[量化] 策略: Embed/Norm=FP16, Q/K=INT8, V/O/FFN=INT4-G128")


# ──────────────── 命令行入口 ────────────────

def main():
    parser = argparse.ArgumentParser(description="FishAI v3 量化工具")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出文件路径")
    parser.add_argument("--format", type=str, default="binary",
                        choices=["binary", "json"],
                        help="输出格式 (默认: binary)")
    parser.add_argument("--no-group-quant", action="store_true",
                        help="不使用分组量化")
    parser.add_argument("--eval", action="store_true",
                        help="运行量化感知评估")
    parser.add_argument("--error-report", action="store_true",
                        help="生成量化误差报告")

    args = parser.parse_args()

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _, step, _ = load_model(args.checkpoint, device=device, load_optimizer=False)

    use_group_quant = not args.no_group_quant

    # 量化误差报告
    if args.error_report:
        errors = compute_quantization_error(model, model.config, use_group_quant=use_group_quant)

        # 保存报告
        report_path = args.output + ".error_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, default=str)
        print(f"[量化] 误差报告 -> {report_path}")

    # 量化感知评估
    if args.eval:
        results = quantization_aware_eval(model, model.config, use_group_quant=use_group_quant)

        eval_path = args.output + ".eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[量化] 评估报告 -> {eval_path}")

    # 导出
    if args.format == "binary":
        export_quantized_weights_binary(model, model.config, args.output, use_group_quant=use_group_quant)
    else:
        export_quantized_weights(model, model.config, args.output, use_group_quant=use_group_quant)


if __name__ == "__main__":
    main()
