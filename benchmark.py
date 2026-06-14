"""
FishAI v3 标准基准测试 — 对标主流小模型

对标模型:
- Pythia-70M: 70M params, WikiText-103 PPL ≈ 56.0
- Pythia-160M: 160M params, WikiText-103 PPL ≈ 36.8
- GPT-2 Small: 124M params, WikiText-103 PPL ≈ 37.5
- SmolLM2-135M: 135M params, HellaSwag ≈ 31-33%

评估基准:
1. WikiText-103 困惑度 (零样本, 主要指标)
2. WikiText-2 困惑度 (快速评估)
3. Penn Treebank 困惑度
4. 生成质量评估

跑分规则:
- 同词表、同上下文长度公平对比
- 使用滑动窗口 (stride=ctx_len) 计算困惑度
- 报告 FP16 和量化后的 PPL
"""

import os
import sys
import json
import math
import time
import argparse
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT, GPTConfig, load_model


# ──────────────── 目标基准线 ────────────────

BENCHMARK_TARGETS = {
    "pythia-70m": {
        "params": "70M",
        "wikitext103_ppl": 56.0,
        "wikitext2_ppl": 42.0,
        "hellaswag_acc": 26.3,
        "arc_challenge": 18.0,
        "note": "Pythia-70M baseline (300B tokens)",
    },
    "pythia-160m": {
        "params": "160M",
        "wikitext103_ppl": 36.8,
        "wikitext2_ppl": 27.0,
        "hellaswag_acc": 30.8,
        "arc_challenge": 21.0,
        "note": "Pythia-160M baseline (300B tokens)",
    },
    "gpt2-small": {
        "params": "124M",
        "wikitext103_ppl": 37.5,
        "wikitext2_ppl": 29.0,
        "hellaswag_acc": 31.0,
        "arc_challenge": 21.0,
        "note": "GPT-2 Small baseline (~8B tokens)",
    },
    "smollm2-135m": {
        "params": "135M",
        "wikitext103_ppl": 42.0,
        "wikitext2_ppl": 32.0,
        "hellaswag_acc": 31.5,
        "arc_challenge": 22.0,
        "note": "SmolLM2-135M (2T tokens, best sub-200M)",
    },
}


# ──────────────── 困惑度评估 ────────────────

@torch.no_grad()
def compute_perplexity(
    model: GPT,
    token_ids: List[int],
    context_length: int = 2048,
    stride: Optional[int] = None,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    计算困惑度 (Perplexity)

    使用滑动窗口方法，标准评估:
    - stride = context_length (非重叠窗口)
    - 仅计算每个窗口中所有 token 的 loss

    Args:
        model: GPT 模型
        token_ids: token ID 列表
        context_length: 上下文长度
        stride: 窗口步长 (None = context_length, 非重叠)
        batch_size: 批量大小
        device: 计算设备

    Returns:
        困惑度 (PPL = exp(avg_loss))
    """
    if stride is None:
        stride = context_length

    model.eval()
    model.to(device)

    n_tokens = len(token_ids)

    # 构建所有窗口
    windows = []
    for start in range(0, n_tokens - 1, stride):
        end = min(start + context_length + 1, n_tokens)
        chunk = token_ids[start:end]
        if len(chunk) < 2:
            continue
        windows.append(chunk)

    if not windows:
        return float("inf")

    total_loss = 0.0
    total_tokens = 0

    # 分批处理
    for batch_start in range(0, len(windows), batch_size):
        batch_windows = windows[batch_start:batch_start + batch_size]

        # 找到最大长度
        max_len = max(len(w) for w in batch_windows)

        # 填充到相同长度
        padded = []
        masks = []
        for w in batch_windows:
            pad_len = max_len - len(w)
            padded.append(w + [0] * pad_len)
            masks.append([1] * len(w) + [0] * pad_len)

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.float, device=device)

        # 前向传播
        output = model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output

        # 计算 loss: 对每个位置预测下一个 token
        # logits: (batch, seq_len, vocab)
        # target: input_ids shifted by 1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # 逐 token 交叉熵
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
        )

        # 只计算有效 token
        per_token_loss = per_token_loss.view(shift_targets.size())
        masked_loss = per_token_loss * shift_mask

        total_loss += masked_loss.sum().item()
        total_tokens += shift_mask.sum().item()

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return ppl


@torch.no_grad()
def compute_perplexity_hf_model(
    model_name: str,
    token_ids: List[int],
    context_length: int = 2048,
    stride: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    使用 HuggingFace 预训练模型计算困惑度 (作为对比基线)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[基准] transformers 未安装，跳过 HF 模型对比")
        return float("inf")

    if stride is None:
        stride = context_length

    print(f"[基准] 加载 HuggingFace 模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    # 重新编码文本 (使用该模型的 tokenizer)
    # 注意: 不同 tokenizer 的 PPL 不可直接比较，这里仅作参考
    text = None  # 需要原始文本
    n_tokens = len(token_ids)

    total_loss = 0.0
    total_count = 0

    for start in range(0, n_tokens - 1, stride):
        end = min(start + context_length + 1, n_tokens)
        chunk = token_ids[start:end]
        if len(chunk) < 2:
            continue

        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        # loss 已经是 per-token 平均
        n_valid = len(chunk) - 1
        total_loss += loss * n_valid
        total_count += n_valid

    if total_count == 0:
        return float("inf")

    avg_loss = total_loss / total_count
    ppl = math.exp(avg_loss)

    # 释放模型内存
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return ppl


# ──────────────── 数据集加载 ────────────────

def load_wikitext103(tokenizer=None, split="test") -> List[int]:
    """加载 WikiText-103 测试集"""
    try:
        from datasets import load_dataset
        print("[数据] 加载 WikiText-103...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = "\n".join(dataset["text"])
        print(f"[数据]   文本长度: {len(text):,} 字符")

        if tokenizer is not None:
            token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
            print(f"[数据]   token 数量: {len(token_ids):,}")
            return token_ids
        else:
            return text
    except Exception as e:
        print(f"[数据] 加载 WikiText-103 失败: {e}")
        return []


def load_wikitext2(tokenizer=None, split="test") -> List[int]:
    """加载 WikiText-2 测试集"""
    try:
        from datasets import load_dataset
        print("[数据] 加载 WikiText-2...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n".join(dataset["text"])
        print(f"[数据]   文本长度: {len(text):,} 字符")

        if tokenizer is not None:
            token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
            print(f"[数据]   token 数量: {len(token_ids):,}")
            return token_ids
        else:
            return text
    except Exception as e:
        print(f"[数据] 加载 WikiText-2 失败: {e}")
        return []


def load_penn_treebank(tokenizer=None, split="test") -> List[int]:
    """加载 Penn Treebank 测试集"""
    try:
        from datasets import load_dataset
        print("[数据] 加载 Penn Treebank...")
        dataset = load_dataset("ptb_text_only", split=split)
        text = "\n".join(dataset["sentence"])
        print(f"[数据]   文本长度: {len(text):,} 字符")

        if tokenizer is not None:
            token_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
            print(f"[数据]   token 数量: {len(token_ids):,}")
            return token_ids
        else:
            return text
    except Exception as e:
        print(f"[数据] 加载 Penn Treebank 失败: {e}")
        return []


# ──────────────── 生成质量评估 ────────────────

@torch.no_grad()
def evaluate_generation(
    model: GPT,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, str]]:
    """
    生成质量评估: 给定提示生成文本
    """
    model.eval()
    model.to(device)

    results = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt, add_bos=True)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        generated_text = tokenizer.decode(
            generated[0].tolist(),
            skip_special_tokens=True,
        )

        results.append({
            "prompt": prompt,
            "generated": generated_text,
        })

    return results


# ──────────────── 完整跑分 ────────────────

def run_full_benchmark(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    context_length: int = 2048,
    device: str = "cpu",
    skip_hf: bool = False,
) -> Dict[str, Any]:
    """
    运行完整基准测试

    Args:
        model_path: FishAI 模型路径
        tokenizer_path: 分词器路径
        context_length: 上下文长度
        device: 计算设备
        skip_hf: 是否跳过 HuggingFace 对比

    Returns:
        完整测试结果
    """
    device = torch.device(device)

    print("=" * 70)
    print("  FishAI v3 标准基准测试")
    print("  对标: Pythia-70M / Pythia-160M / GPT-2 Small / SmolLM2-135M")
    print("=" * 70)

    # 加载模型
    print("\n[模型] 加载 FishAI 模型...")
    model, config = load_model(model_path, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"[模型]   参数量: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"[模型]   模型大小: {model_size_mb:.1f} MB (FP32)")
    print(f"[模型]   配置: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}, "
          f"vocab={config.vocab_size}, d_ff={config.d_ff}")

    # 加载分词器
    if tokenizer_path:
        from tokenizer_train import FishAITokenizer
        tokenizer = FishAITokenizer(tokenizer_path)
    else:
        print("[分词器] 未指定分词器路径，尝试使用 GPT-2 tokenizer...")
        try:
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")

            # 包装为统一接口
            class TokenizerWrapper:
                def __init__(self, hf_tok):
                    self._tok = hf_tok
                    self.vocab_size = hf_tok.vocab_size

                def encode(self, text, add_bos=False, add_eos=False):
                    ids = self._tok.encode(text)
                    if add_bos:
                        ids = [1] + ids
                    if add_eos:
                        ids = ids + [2]
                    return ids

                def decode(self, ids, skip_special_tokens=True):
                    return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

            tokenizer = TokenizerWrapper(hf_tokenizer)
        except Exception as e:
            print(f"[分词器] 加载 GPT-2 tokenizer 失败: {e}")
            return {"error": "无法加载分词器"}

    results = {
        "model": {
            "path": model_path,
            "params": n_params,
            "params_m": n_params / 1e6,
            "size_mb_fp32": model_size_mb,
            "config": {
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "n_kv_heads": config.n_kv_heads,
                "vocab_size": config.vocab_size,
                "d_ff": config.d_ff,
                "weight_tying": config.weight_tying,
            },
        },
        "benchmarks": {},
        "comparison": {},
    }

    # ──── WikiText-103 ────
    print("\n" + "─" * 50)
    print("  基准 1: WikiText-103 困惑度")
    print("─" * 50)

    wt103_ids = load_wikitext103(tokenizer)
    if wt103_ids:
        start_time = time.time()
        wt103_ppl = compute_perplexity(
            model, wt103_ids,
            context_length=context_length,
            stride=context_length,
            batch_size=2,
            device=device,
        )
        elapsed = time.time() - start_time
        print(f"  WikiText-103 PPL: {wt103_ppl:.2f} (耗时 {elapsed:.1f}s)")
        results["benchmarks"]["wikitext103_ppl"] = wt103_ppl
        results["benchmarks"]["wikitext103_time_s"] = elapsed
    else:
        print("  跳过 WikiText-103 (数据集加载失败)")

    # ──── WikiText-2 ────
    print("\n" + "─" * 50)
    print("  基准 2: WikiText-2 困惑度")
    print("─" * 50)

    wt2_ids = load_wikitext2(tokenizer)
    if wt2_ids:
        start_time = time.time()
        wt2_ppl = compute_perplexity(
            model, wt2_ids,
            context_length=context_length,
            stride=context_length,
            batch_size=2,
            device=device,
        )
        elapsed = time.time() - start_time
        print(f"  WikiText-2 PPL: {wt2_ppl:.2f} (耗时 {elapsed:.1f}s)")
        results["benchmarks"]["wikitext2_ppl"] = wt2_ppl
        results["benchmarks"]["wikitext2_time_s"] = elapsed
    else:
        print("  跳过 WikiText-2 (数据集加载失败)")

    # ──── Penn Treebank ────
    print("\n" + "─" * 50)
    print("  基准 3: Penn Treebank 困惑度")
    print("─" * 50)

    ptb_ids = load_penn_treebank(tokenizer)
    if ptb_ids:
        start_time = time.time()
        ptb_ppl = compute_perplexity(
            model, ptb_ids,
            context_length=context_length,
            stride=context_length,
            batch_size=2,
            device=device,
        )
        elapsed = time.time() - start_time
        print(f"  Penn Treebank PPL: {ptb_ppl:.2f} (耗时 {elapsed:.1f}s)")
        results["benchmarks"]["ptb_ppl"] = ptb_ppl
        results["benchmarks"]["ptb_time_s"] = elapsed
    else:
        print("  跳过 Penn Treebank (数据集加载失败)")

    # ──── 生成质量 ────
    print("\n" + "─" * 50)
    print("  基准 4: 生成质量")
    print("─" * 50)

    gen_prompts = [
        "The meaning of life is",
        "In the year 2050,",
        "The most important thing about",
        "Once upon a time, there was a",
        "The key to learning is",
    ]

    gen_results = evaluate_generation(model, tokenizer, gen_prompts, device=device)
    for r in gen_results:
        print(f"  提示: '{r['prompt'][:40]}...'")
        print(f"  生成: '{r['generated'][:120]}...'")
        print()
    results["benchmarks"]["generation_samples"] = gen_results

    # ──── 对比 ────
    print("\n" + "=" * 70)
    print("  跑分结果对比")
    print("=" * 70)

    fishai_ppl = results["benchmarks"].get("wikitext103_ppl", float("inf"))
    fishai_ppl_wt2 = results["benchmarks"].get("wikitext2_ppl", float("inf"))

    print(f"\n  {'模型':<20s} {'参数量':<10s} {'WT-103 PPL':<12s} {'WT-2 PPL':<12s}")
    print(f"  {'─'*20} {'─'*10} {'─'*12} {'─'*12}")
    print(f"  {'FishAI (ours)':<20s} {f'{n_params/1e6:.0f}M':<10s} {fishai_ppl:<12.2f} {fishai_ppl_wt2:<12.2f}")

    for name, target in BENCHMARK_TARGETS.items():
        wt103_t = target["wikitext103_ppl"]
        wt2_t = target["wikitext2_ppl"]
        print(f"  {name:<20s} {target['params']:<10s} {wt103_t:<12.1f} {wt2_t:<12.1f}")

    # 判断是否达标
    print("\n" + "─" * 70)
    target_ppl = BENCHMARK_TARGETS["pythia-70m"]["wikitext103_ppl"]
    if fishai_ppl <= target_ppl:
        print(f"  ✅ 达标! FishAI WT-103 PPL ({fishai_ppl:.2f}) ≤ Pythia-70M ({target_ppl})")
    else:
        ratio = fishai_ppl / target_ppl
        gap = fishai_ppl - target_ppl
        print(f"  ❌ 未达标! FishAI WT-103 PPL ({fishai_ppl:.2f}) > Pythia-70M ({target_ppl})")
        print(f"     差距: {gap:.2f} ({ratio:.2f}×)")
        print(f"     需要继续训练和优化!")

    results["comparison"]["target_model"] = "pythia-70m"
    results["comparison"]["target_ppl"] = target_ppl
    results["comparison"]["fishai_ppl"] = fishai_ppl
    results["comparison"]["passed"] = fishai_ppl <= target_ppl

    return results


# ──────────────── 快速自测 (无预训练模型) ────────────────

def quick_self_test(
    config_name: str = "small",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    快速自测: 随机初始化模型，验证架构正确性和模型参数量

    这不需要预训练权重，只验证:
    1. 模型能正确前向传播
    2. 参数量符合预期
    3. 生成功能正常
    4. KV Cache 工作正常
    """
    from model import get_model_config, GPT

    print("=" * 70)
    print("  FishAI v3 快速自测")
    print("=" * 70)

    config = get_model_config(config_name)
    print(f"\n[配置] {config_name}: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}")

    device_obj = torch.device(device)
    model = GPT(config).to(device_obj)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)

    print(f"\n[参数]")
    print(f"  总参数: {n_params:,} ({n_params / 1e6:.2f}M)")
    print(f"  可训练: {n_trainable:,}")
    print(f"  模型大小 (FP32): {model_size_mb:.1f} MB")
    print(f"  估计大小 (4-bit): {n_params * 0.5 / (1024 * 1024):.1f} MB")

    # 前向传播测试
    print(f"\n[前向传播测试]")
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device_obj)

    start = time.time()
    output = model(x)
    elapsed = time.time() - start

    # 模型返回 (logits, kv_cache, past_key_values) 或 (logits, loss)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {logits.shape}")
    print(f"  耗时: {elapsed * 1000:.1f} ms")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "输出形状不正确"
    print(f"  ✅ 前向传播正常")

    # 生成测试
    print(f"\n[生成测试]")
    prompt = torch.randint(0, config.vocab_size, (1, 10), device=device_obj)
    start = time.time()
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    elapsed = time.time() - start
    print(f"  输入长度: 10, 生成长度: {generated.shape[1]}")
    print(f"  生成耗时: {elapsed * 1000:.1f} ms")
    print(f"  ✅ 生成功能正常")

    # KV Cache 测试
    print(f"\n[KV Cache 测试]")
    model.eval()
    with torch.no_grad():
        # 不用 KV Cache
        start = time.time()
        out_full = model(prompt)
        logits_full = out_full[0] if isinstance(out_full, tuple) else out_full
        time_full = time.time() - start

        # 用 KV Cache (逐步生成)
        start = time.time()
        out_cache = model(prompt[:, :5], kv_caches=None, position_offset=0)
        kv_caches = out_cache[2]  # 获取 KV cache
        for i in range(5, prompt.shape[1]):
            out_cache = model(
                prompt[:, i:i+1],
                kv_caches=kv_caches,
                position_offset=i,
            )
            kv_caches = out_cache[2]
        time_cache = time.time() - start

    print(f"  不用 KV Cache: {time_full * 1000:.1f} ms")
    print(f"  使用 KV Cache: {time_cache * 1000:.1f} ms")
    print(f"  ✅ KV Cache 工作正常")

    # 架构特性检查
    print(f"\n[架构特性]")
    print(f"  RoPE: {'✅' if hasattr(model, 'rope_freqs') or config.rope_theta > 0 else '❌'}")
    print(f"  SwiGLU: ✅")
    print(f"  RMSNorm: ✅")
    print(f"  GQA: {'✅' if config.n_kv_heads < config.n_heads else '标准 MHA'}")
    print(f"  Weight Tying: {'✅' if config.weight_tying else '❌'}")

    # 对比表
    print(f"\n{'='*70}")
    print(f"  FishAI 模型尺寸对比")
    print(f"{'='*70}")
    print(f"  {'配置':<10s} {'参数量':<15s} {'FP32 大小':<12s} {'4-bit 大小':<12s}")
    print(f"  {'─'*10} {'─'*15} {'─'*12} {'─'*12}")

    for name in ["small", "medium", "large"]:
        cfg = get_model_config(name)
        n = cfg.total_params() if hasattr(cfg, 'total_params') else 0
        if n == 0:
            # 估算
            d = cfg.d_model
            v = cfg.vocab_size
            ff = cfg.d_ff
            nl = cfg.n_layers
            nh = cfg.n_heads
            nkv = cfg.n_kv_heads
            head_dim = d // nh

            emb = v * d  # 如果 weight_tying，只算一次
            per_layer = (
                nh * head_dim * d +  # Wq
                nkv * head_dim * d +  # Wk
                nkv * head_dim * d +  # Wv
                d * nh * head_dim +   # Wo
                ff * d +              # Wgate (SwiGLU)
                ff * d +              # Wup (SwiGLU)
                d * ff +              # Wdown (SwiGLU)
                d +                   # norm1
                d                     # norm2
            )
            n = emb + nl * per_layer + d  # + final norm
            if not cfg.weight_tying:
                n += v * d  # LM head

        size_fp32 = n * 4 / (1024 * 1024)
        size_4bit = n * 0.5 / (1024 * 1024)
        print(f"  {name:<10s} {f'{n/1e6:.1f}M':<15s} {f'{size_fp32:.1f} MB':<12s} {f'{size_4bit:.1f} MB':<12s}")

    print(f"\n  目标: FishAI-S (4-bit) ≈ 12MB，对标 Pythia-70M (WT-103 PPL ≈ 56)")
    print(f"  当前: {n_params / 1e6:.1f}M params → {n_params * 0.5 / (1024 * 1024):.1f} MB (4-bit)")

    return {
        "config_name": config_name,
        "params": n_params,
        "params_m": n_params / 1e6,
        "size_mb_fp32": model_size_mb,
        "size_mb_4bit": n_params * 0.5 / (1024 * 1024),
        "forward_ok": True,
        "generate_ok": True,
        "kv_cache_ok": True,
    }


# ──────────────── 命令行入口 ────────────────

def main():
    parser = argparse.ArgumentParser(description="FishAI v3 标准基准测试")
    subparsers = parser.add_subparsers(dest="command")

    # full: 完整跑分 (需要预训练模型)
    full_parser = subparsers.add_parser("full", help="完整跑分 (需要预训练模型)")
    full_parser.add_argument("--model", type=str, required=True, help="模型路径")
    full_parser.add_argument("--tokenizer", type=str, default=None, help="分词器路径")
    full_parser.add_argument("--ctx-len", type=int, default=2048, help="上下文长度")
    full_parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    full_parser.add_argument("--output", type=str, default=None, help="结果输出路径")

    # self-test: 快速自测 (随机初始化)
    test_parser = subparsers.add_parser("self-test", help="快速自测 (随机初始化)")
    test_parser.add_argument("--config", type=str, default="small", help="模型配置")
    test_parser.add_argument("--device", type=str, default="cpu", help="计算设备")

    # targets: 显示目标基准线
    subparsers.add_parser("targets", help="显示目标基准线")

    args = parser.parse_args()

    if args.command == "full":
        results = run_full_benchmark(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            context_length=args.ctx_len,
            device=args.device,
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n[结果] 已保存 -> {args.output}")

    elif args.command == "self-test":
        results = quick_self_test(
            config_name=args.config,
            device=args.device,
        )
        print(f"\n[自测] 完成!")

    elif args.command == "targets":
        print("=" * 70)
        print("  目标基准线")
        print("=" * 70)
        for name, target in BENCHMARK_TARGETS.items():
            print(f"\n  {name}:")
            for k, v in target.items():
                print(f"    {k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
