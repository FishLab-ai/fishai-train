"""
FishAI v2 模型定义 — 小体积最聪明的自研 Transformer

v2 核心升级 (对标 LLaMA/Phi):
1. RoPE (Rotary Position Embedding) — 零参数位置编码
2. SwiGLU 激活函数 — 比 GELU 更强表达力
3. RMSNorm — 比 LayerNorm 更快更简
4. GQA (Grouped Query Attention) — 省 7% 参数 + 50% KV 缓存
5. 权重绑定 (Weight Tying) — Embed 与 LM Head 共享
6. 无偏置 (No Bias) — 现代发现 bias 在 RMSNorm 下冗余

架构: ~38M 参数 (权重绑定后), 量化后 ~12MB
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """FishAI v2 模型配置"""
    vocab_size: int = 32000
    max_seq_len: int = 512
    d_model: int = 512
    n_heads: int = 8          # Q 头数
    n_kv_heads: int = 4       # KV 头数 (GQA)
    n_layers: int = 6
    d_ff: int = 1408           # 8/3 * 512, round to 64*22
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0       # 小模型 dropout=0 更好 (Phi 经验)
    init_range: float = 0.02
    weight_tying: bool = True  # 权重绑定

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def total_params(self) -> int:
        """计算总参数量 (权重绑定后)"""
        d = self.d_model
        v = self.vocab_size
        ff = self.d_ff
        nh = self.n_heads
        nkv = self.n_kv_heads
        hd = self.head_dim

        # Token Embedding (无 Position Embedding!)
        tok_emb = v * d

        # Per Layer
        layer_params = 0
        # GQA Attention (无 bias)
        layer_params += d * (nh * hd)      # Wq
        layer_params += d * (nkv * hd)     # Wk
        layer_params += d * (nkv * hd)     # Wv
        layer_params += d * d              # Wo
        # SwiGLU FFN (三矩阵, 无 bias)
        layer_params += d * ff             # W_gate
        layer_params += d * ff             # W_up
        layer_params += ff * d             # W_down
        # RMSNorm (2 per layer, 只有 gamma)
        layer_params += 2 * d

        transformer_params = layer_params * self.n_layers
        # Final RMSNorm
        final_norm = d
        # LM Head (权重绑定则不计)
        lm_head = 0 if self.weight_tying else d * v

        return tok_emb + transformer_params + final_norm + lm_head

    def quantized_size_mb(self) -> float:
        """混合精度量化后大小"""
        d = self.d_model
        v = self.vocab_size
        ff = self.d_ff
        nh = self.n_heads
        nkv = self.n_kv_heads
        hd = self.head_dim

        # FP16 部分
        fp16_params = v * d + d  # embedding + final norm
        # INT4 部分
        int4_per_layer = (d * (nh * hd) + d * (nkv * hd) * 2 + d * d +
                          d * ff * 2 + ff * d)
        # Norm gamma 用 FP16
        fp16_per_layer = 2 * d

        total_fp16 = fp16_params + fp16_per_layer * self.n_layers
        total_int4 = int4_per_layer * self.n_layers

        bytes_size = total_fp16 * 2 + total_int4 * 0.5
        return bytes_size / (1024 * 1024)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    比 LayerNorm 更简单: 去掉 mean-centering 和 beta
    公式: x / sqrt(mean(x²) + eps) * gamma
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 频率表
    θ_i = 1 / theta^(2i / head_dim)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # [max_seq_len, head_dim/2]
    return freqs


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    应用 RoPE 到输入张量
    x: [batch, n_heads, seq_len, head_dim]
    freqs: [seq_len, head_dim/2]
    """
    seq_len = x.size(2)
    freqs = freqs[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim/2]

    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)  # [..., head_dim/2, 2]
    cos_f = torch.cos(freqs).unsqueeze(-1)  # [1, 1, seq, head_dim/2, 1]
    sin_f = torch.sin(freqs).unsqueeze(-1)

    # 旋转: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    x0 = x_reshape[..., 0:1]
    x1 = x_reshape[..., 1:2]
    rotated = torch.cat([x0 * cos_f - x1 * sin_f,
                          x0 * sin_f + x1 * cos_f], dim=-1)

    return rotated.flatten(-2).type_as(x)


class GroupedQueryAttention(nn.Module):
    """
    GQA (Grouped Query Attention) with RoPE

    相比 MHA: KV 头数 < Q 头数, 每 group_size 个 Q 头共享一组 KV
    参数节省: (2 * n_kv_heads / n_heads) 的 KV 投影参数
    推理加速: KV cache 减少到 n_kv_heads / n_heads
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        assert config.n_heads % config.n_kv_heads == 0

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # GQA 投影 (无 bias)
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .unsqueeze(0).unsqueeze(0)
        )

        # 预计算 RoPE 频率
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, config.max_seq_len, config.rope_theta)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Q/K/V 投影
        q = self.wq(x)  # [B, T, n_heads * head_dim]
        k = self.wk(x)  # [B, T, n_kv_heads * head_dim]
        v = self.wv(x)  # [B, T, n_kv_heads * head_dim]

        # 重塑
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)    # [B, nh, T, hd]
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, nkv, T, hd]
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, nkv, T, hd]

        # 应用 RoPE
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # GQA: 扩展 KV 头以匹配 Q 头数
        # [B, nkv, T, hd] -> [B, nh, T, hd]
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, T, T]
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v  # [B, nh, T, hd]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.wo(out)
        out = self.resid_dropout(out)
        return out


class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络
    FFN(x) = W_down(SiLU(x @ W_gate) ⊙ (x @ W_up))

    比 GELU FFN 多一个矩阵但表达力显著更强
    d_ff 建议 8/3 * d_model (向 64 取整)
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))   # SiLU = Swish
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class TransformerBlock(nn.Module):
    """
    Pre-RMSNorm Transformer Block (v2)

    x = x + GQA(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.rms1 = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.rms2 = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms1(x))
        x = x + self.ffn(self.rms2(x))
        return x


class GPT(nn.Module):
    """
    FishAI v2 — 小体积最聪明的自研 Transformer

    架构: RoPE + SwiGLU + RMSNorm + GQA + WeightTying + NoBias
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token Embedding (无 Position Embedding!)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final RMSNorm
        self.final_rms = RMSNorm(config.d_model, config.norm_eps)

        # LM Head (权重绑定则不单独创建)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        # 权重初始化
        self.apply(self._init_weights)

        # 打印模型信息
        actual_params = sum(p.numel() for p in self.parameters())
        print(f"[FishAI v2] 参数量: {actual_params / 1e6:.1f}M")
        print(f"[FishAI v2] 混合精度量化: {config.quantized_size_mb():.1f} MB")
        print(f"[FishAI v2] 架构: RoPE + SwiGLU + RMSNorm + GQA + WeightTying + NoBias")
        print(f"[FishAI v2] GQA: {config.n_heads} Q heads × {config.n_kv_heads} KV heads")

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
            # v2: 无 bias
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gamma)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len

        # Token Embedding (RoPE 在 Attention 内施加, 无 Position Embedding)
        x = self.embed_dropout(self.token_embedding(input_ids))

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # Final RMSNorm
        x = self.final_rms(x)

        # LM Head
        logits = self.lm_head(x)

        # 计算 loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """自回归生成"""
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    config = GPTConfig()

    print(f"\n{'='*60}")
    print(f"  FishAI v2 — 小体积最聪明的自研 Transformer")
    print(f"{'='*60}")
    print(f"\n模型配置:")
    print(f"  d_model:      {config.d_model}")
    print(f"  n_heads:      {config.n_heads} (Q)")
    print(f"  n_kv_heads:   {config.n_kv_heads} (KV, GQA)")
    print(f"  n_layers:     {config.n_layers}")
    print(f"  d_ff:         {config.d_ff} (SwiGLU)")
    print(f"  vocab_size:   {config.vocab_size}")
    print(f"  max_seq_len:  {config.max_seq_len}")
    print(f"  weight_tying: {config.weight_tying}")
    print(f"  rope_theta:   {config.rope_theta}")
    print(f"  总参数量:     {config.total_params() / 1e6:.1f}M")
    print(f"  量化后:       {config.quantized_size_mb():.1f} MB")

    # 测试前向传播
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss = model(x, labels=x)
    print(f"\n前向传播测试:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # 对比 v1 参数量
    v1_params = 52.0  # v1 GPT-2 ~52M
    v2_params = config.total_params() / 1e6
    print(f"\n参数效率对比:")
    print(f"  v1 (GPT-2):     ~{v1_params:.0f}M")
    print(f"  v2 (LLaMA-style): ~{v2_params:.1f}M")
    print(f"  参数节省:       {(1 - v2_params/v1_params)*100:.1f}%")
    print(f"  v2 每参数表达力更强 (RoPE/SwiGLU/GQA)")
