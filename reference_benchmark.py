"""
FishAI v3 对标跑分 — 使用 HuggingFace 预训练模型获取参考基线

直接加载 Pythia-70M / GPT-2 Small 计算 WikiText PPL，
作为 FishAI 训练目标的对标。
"""

import os
import sys
import math
import time
import json

def run_reference_benchmarks():
    """运行参考模型的 PPL 基准测试"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    device = torch.device("cpu")
    ctx_len = 1024  # 使用 1024 上下文长度 (速度考虑)
    stride = ctx_len

    models_to_test = {
        "gpt2": {"name": "GPT-2 Small (124M)", "expected_ppl": 37.5},
        "EleutherAI/pythia-70m": {"name": "Pythia-70M", "expected_ppl": 56.0},
        "EleutherAI/pythia-160m": {"name": "Pythia-160M", "expected_ppl": 36.8},
    }

    results = {}

    for model_id, info in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"  评估: {info['name']} ({model_id})")
        print(f"{'='*60}")

        try:
            print(f"  加载模型...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
            model.eval()

            n_params = sum(p.numel() for p in model.parameters())
            print(f"  参数量: {n_params:,} ({n_params/1e6:.1f}M)")

            # 加载 WikiText-2 (比 103 小，快速评估)
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n".join(dataset["text"])
            token_ids = tokenizer.encode(text)
            print(f"  Token 数量: {len(token_ids):,}")

            # 计算 PPL
            print(f"  计算 PPL (ctx_len={ctx_len}, stride={stride})...")
            total_loss = 0.0
            total_tokens = 0
            n_windows = 0

            start_time = time.time()
            with torch.no_grad():
                for start in range(0, len(token_ids) - 1, stride):
                    end = min(start + ctx_len + 1, len(token_ids))
                    chunk = token_ids[start:end]
                    if len(chunk) < 2:
                        continue

                    input_ids = torch.tensor([chunk], dtype=torch.long)
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss.item()

                    n_valid = len(chunk) - 1
                    total_loss += loss * n_valid
                    total_tokens += n_valid
                    n_windows += 1

                    if n_windows % 20 == 0:
                        current_ppl = math.exp(total_loss / total_tokens)
                        print(f"    窗口 {n_windows}: 当前 PPL = {current_ppl:.2f}")

            elapsed = time.time() - start_time
            avg_loss = total_loss / total_tokens
            ppl = math.exp(avg_loss)

            print(f"\n  结果:")
            print(f"    平均 Loss: {avg_loss:.4f}")
            print(f"    PPL: {ppl:.2f}")
            print(f"    预期 PPL: {info['expected_ppl']}")
            print(f"    耗时: {elapsed:.1f}s")

            results[model_id] = {
                "name": info["name"],
                "params_m": n_params / 1e6,
                "wikitext2_ppl": ppl,
                "expected_ppl": info["expected_ppl"],
                "ctx_len": ctx_len,
                "n_windows": n_windows,
            }

            # 释放模型
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  评估失败: {e}")
            results[model_id] = {"error": str(e)}

    # 汇总
    print(f"\n{'='*60}")
    print(f"  参考基线汇总")
    print(f"{'='*60}")
    print(f"  {'模型':<25s} {'参数量':<10s} {'WT-2 PPL':<12s} {'预期 PPL':<12s}")
    print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*12}")

    for model_id, r in results.items():
        if "error" not in r:
            name_str = r['name']
            params_str = f"{r['params_m']:.0f}M"
            ppl_str = f"{r['wikitext2_ppl']:.2f}"
            expected_str = f"{r['expected_ppl']:.1f}"
            print(f"  {name_str:<25s} {params_str:<10s} {ppl_str:<12s} {expected_str:<12s}")
        else:
            print(f"  {model_id:<25s} 评估失败")

    # FishAI 目标
    print(f"\n  FishAI-Small 目标: WT-2 PPL ≤ {results.get('EleutherAI/pythia-70m', {}).get('wikitext2_ppl', 42.0):.1f}")

    return results


if __name__ == "__main__":
    results = run_reference_benchmarks()
    with open("/tmp/reference_benchmarks.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: /tmp/reference_benchmarks.json")
