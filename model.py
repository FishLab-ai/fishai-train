"""
FishAI v3 模型定义 — 小体积最聪明的自研 Transformer

v3 核心升级 (对标 LLaMA-3/Phi-3):
1. RoPE (Rotary Position Embedding) — 零参数位置编码，共享频率缓冲区
2. SwiGLU 激活函数 — 比 GELU 更强表达力
3. RMSNorm — 比 LayerNorm 更快更简
4. GQA (Grouped Query Attention) — 省 KV 缓存，加速推理
5. 权重绑定 (Weight Tying) — Embed 与 LM Head 共享
6. 无偏置 (No Bias) — 现代发现 bias 在 RMSNorm 下冗余
7. Flash Attention — 使用 torch.nn.functional.scaled_dot_product_attention
8. 多尺寸配置 — small (~34M) / medium (~400M) / large (~1.5B)
9. KV Cache — 推理时增量计算，避免重复计算
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────── 模型配置 ────────────────

@dataclass
class GPTConfig:
    """FishAI v3 模型配置"""
    vocab_size: int = 32000
    max_seq_len: int = 2048
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
        """每个 KV 头服务多少个 Q 头"""
        return self.n_heads // self.n_kv_heads

    def total_params(self) -> int:
        """计算总参数量 (权重绑定后)"""
        d = self.d_model
        v = self.vocab_size
        ff = self.d_ff
        nh = self.n_heads
        nkv = self.n_kv_heads
        hd = self.head_dim

        # Token Embedding
        tok_emb = v * d

        # 每层参数
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
        """混合精度量化后大小 (FP16 embedding/norm + INT4 其余)"""
        d = self.d_model
        v = self.vocab_size
        ff = self.d_ff
        nh = self.n_heads
        nkv = self.n_kv_heads
        hd = self.head_dim

        # FP16 部分: embedding + final norm + per-layer norms
        fp16_params = v * d + d
        fp16_per_layer = 2 * d
        total_fp16 = fp16_params + fp16_per_layer * self.n_layers

        # INT4 部分: 所有线性层
        int4_per_layer = (
            d * (nh * hd) +           # Wq
            d * (nkv * hd) * 2 +      # Wk, Wv
            d * d +                    # Wo
            d * ff * 2 + ff * d        # W_gate, W_up, W_down
        )
        total_int4 = int4_per_layer * self.n_layers

        bytes_size = total_fp16 * 2 + total_int4 * 0.5
        return bytes_size / (1024 * 1024)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
            "dropout": self.dropout,
            "init_range": self.init_range,
            "weight_tying": self.weight_tying,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GPTConfig":
        """从字典反序列化"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def get_model_config(size: str = "small") -> GPTConfig:
    """
    获取预设模型配置

    Args:
        size: "small" (~34M), "medium" (~400M), "large" (~1.5B)

    Returns:
        GPTConfig 实例
    """
    configs = {
        "small": GPTConfig(
            vocab_size=32000,
            max_seq_len=2048,
            d_model=512,
            n_heads=8,
            n_kv_heads=4,
            n_layers=6,
            d_ff=1408,         # 8/3 * 512, round to 64*22
            rope_theta=10000.0,
            dropout=0.0,
            weight_tying=True,
        ),
        "medium": GPTConfig(
            vocab_size=32000,
            max_seq_len=4096,
            d_model=896,
            n_heads=14,
            n_kv_heads=2,
            n_layers=24,
            d_ff=4864,         # 8/3 * 896, round to 64*76
            rope_theta=10000.0,
            dropout=0.0,
            weight_tying=True,
        ),
        "large": GPTConfig(
            vocab_size=32000,
            max_seq_len=4096,
            d_model=1536,
            n_heads=12,
            n_kv_heads=4,
            n_layers=28,
            d_ff=8960,         # 8/3 * 1536, round to 64*140
            rope_theta=10000.0,
            dropout=0.0,
            weight_tying=False,  # 大模型通常不绑定权重
        ),
    }

    if size not in configs:
        raise ValueError(
            f"未知模型大小 '{size}'，可选: {list(configs.keys())}"
        )

    config = configs[size]
    actual_params = config.total_params()
    print(f"[FishAI v3] 模型大小: {size}")
    print(f"[FishAI v3] 参数量: {actual_params / 1e6:.1f}M")
    print(f"[FishAI v3] 混合精度量化: {config.quantized_size_mb():.1f} MB")
    return config


# ──────────────── 基础组件 ────────────────

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

    def extra_repr(self) -> str:
        return f"{self.gamma.size(0)}, eps={self.eps}"


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 频率表 (cos 和 sin 分开存储，避免重复计算)

    Args:
        head_dim: 每个注意力头的维度
        max_seq_len: 最大序列长度
        theta: RoPE 基础频率

    Returns:
        (cos_freqs, sin_freqs): 各为 [max_seq_len, head_dim/2] 形状
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)  # [max_seq_len, head_dim/2]

    cos_freqs = torch.cos(freqs)  # [max_seq_len, head_dim/2]
    sin_freqs = torch.sin(freqs)  # [max_seq_len, head_dim/2]
    return cos_freqs, sin_freqs


def apply_rope(
    x: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    position_offset: int = 0,
) -> torch.Tensor:
    """
    应用 RoPE 到输入张量

    Args:
        x: [batch, n_heads, seq_len, head_dim]
        cos_freqs: [max_seq_len, head_dim/2]
        sin_freqs: [max_seq_len, head_dim/2]
        position_offset: 位置偏移 (用于 KV cache 推理)

    Returns:
        旋转后的张量，形状同 x
    """
    seq_len = x.size(2)
    pos = position_offset

    # 取出当前位置对应的频率
    cos_f = cos_freqs[pos:pos + seq_len]  # [seq_len, head_dim/2]
    sin_f = sin_freqs[pos:pos + seq_len]  # [seq_len, head_dim/2]

    # 广播到 [1, 1, seq_len, head_dim/2, 1]
    cos_f = cos_f.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, seq, head_dim/2, 1]
    sin_f = sin_f.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, seq, head_dim/2, 1]

    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)  # [..., head_dim/2, 2]

    x0 = x_reshape[..., 0:1]  # [..., head_dim/2, 1]
    x1 = x_reshape[..., 1:2]

    # 旋转: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    rotated = torch.cat([x0 * cos_f - x1 * sin_f,
                          x0 * sin_f + x1 * cos_f], dim=-1)

    return rotated.flatten(-2).type_as(x)


# ──────────────── 注意力层 ────────────────

class GroupedQueryAttention(nn.Module):
    """
    GQA (Grouped Query Attention) with RoPE & Flash Attention

    相比 MHA: KV 头数 < Q 头数, 每 group_size 个 Q 头共享一组 KV
    参数节省: (2 * n_kv_heads / n_heads) 的 KV 投影参数
    推理加速: KV cache 减少到 n_kv_heads / n_heads
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, \
            f"d_model ({config.d_model}) 必须被 n_heads ({config.n_heads}) 整除"
        assert config.n_heads % config.n_kv_heads == 0, \
            f"n_heads ({config.n_heads}) 必须被 n_kv_heads ({config.n_kv_heads}) 整除"

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

    def forward(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_flash: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Args:
            x: [batch, seq_len, d_model]
            cos_freqs: [max_seq_len, head_dim/2] 共享的 RoPE cos 频率
            sin_freqs: [max_seq_len, head_dim/2] 共享的 RoPE sin 频率
            kv_cache: 可选的 KV 缓存 (用于推理)
            position_offset: 位置偏移量 (用于 KV cache)
            use_flash: 是否使用 Flash Attention

        Returns:
            (输出张量, 更新后的 KV 缓存)
        """
        B, T, C = x.size()

        # Q/K/V 投影
        q = self.wq(x)  # [B, T, n_heads * head_dim]
        k = self.wk(x)  # [B, T, n_kv_heads * head_dim]
        v = self.wv(x)  # [B, T, n_kv_heads * head_dim]

        # 重塑为多头形式
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)    # [B, nh, T, hd]
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, nkv, T, hd]
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, nkv, T, hd]

        # 应用 RoPE
        q = apply_rope(q, cos_freqs, sin_freqs, position_offset)
        k = apply_rope(k, cos_freqs, sin_freqs, position_offset)

        # KV Cache: 拼接历史 KV
        new_kv_cache = None
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)  # [B, nkv, T_prev+T, hd]
            v = torch.cat([v_prev, v], dim=2)
        new_kv_cache = (k, v)

        # GQA: 扩展 KV 头以匹配 Q 头数
        # [B, nkv, T_total, hd] -> [B, nh, T_total, hd]
        k_expanded = k.repeat_interleave(self.n_groups, dim=1)
        v_expanded = v.repeat_interleave(self.n_groups, dim=1)

        # Attention 计算
        if use_flash and kv_cache is None:
            # 使用 Flash Attention (仅训练时，需要完整序列)
            attn_output = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 手动计算 (用于 KV cache 推理或 Flash Attention 不可用时)
            S = k_expanded.size(2)  # 总序列长度 (含 cache)
            attn = (q @ k_expanded.transpose(-2, -1)) * self.scale  # [B, nh, T, S]

            # 因果掩码
            if kv_cache is not None:
                # 推理时: 只需掩码当前 query 对未来 key 的注意力
                causal_mask = torch.tril(
                    torch.ones(T, S, device=x.device, dtype=torch.bool),
                    diagonal=S - T,
                )
                attn = attn.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                causal_mask = torch.tril(
                    torch.ones(T, T, device=x.device, dtype=torch.bool)
                )
                attn = attn.masked_fill(
                    ~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf')
                )

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            attn_output = attn @ v_expanded  # [B, nh, T, hd]

        # 合并头
        out = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.wo(out)
        out = self.resid_dropout(out)

        return out, new_kv_cache


# ──────────────── FFN 层 ────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络
    FFN(x) = W_down(SiLU(x @ W_gate) * (x @ W_up))

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


# ──────────────── Transformer Block ────────────────

class TransformerBlock(nn.Module):
    """
    Pre-RMSNorm Transformer Block (v3)

    x = x + GQA(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.rms1 = RMSNorm(config.d_model, config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.rms2 = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_offset: int = 0,
        use_flash: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv_cache = self.attn(
            self.rms1(x), cos_freqs, sin_freqs,
            kv_cache=kv_cache, position_offset=position_offset,
            use_flash=use_flash,
        )
        x = x + attn_out
        x = x + self.ffn(self.rms2(x))
        return x, new_kv_cache


# ──────────────── 主模型 ────────────────

class GPT(nn.Module):
    """
    FishAI v3 — 小体积最聪明的自研 Transformer

    架构: RoPE + SwiGLU + RMSNorm + GQA + WeightTying + NoBias + FlashAttn
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token Embedding (无 Position Embedding — RoPE 在 Attention 内施加)
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

        # 预计算共享 RoPE 频率缓冲区 (不再在每个 Attention 层重复存储)
        cos_freqs, sin_freqs = precompute_rope_freqs(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("rope_cos_freqs", cos_freqs, persistent=False)
        self.register_buffer("rope_sin_freqs", sin_freqs, persistent=False)

        # 权重初始化
        self.apply(self._init_weights)

        # 打印模型信息
        actual_params = sum(p.numel() for p in self.parameters())
        print(f"[FishAI v3] 参数量: {actual_params / 1e6:.1f}M")
        print(f"[FishAI v3] 混合精度量化: {config.quantized_size_mb():.1f} MB")
        print(f"[FishAI v3] 架构: RoPE + SwiGLU + RMSNorm + GQA + WeightTying + FlashAttn")
        print(f"[FishAI v3] GQA: {config.n_heads} Q heads x {config.n_kv_heads} KV heads "
              f"(每组 {config.n_groups} 个 Q 共享 1 个 KV)")

    def _init_weights(self, module: nn.Module) -> None:
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.gamma)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List]]:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len] 输入 token IDs
            labels: [batch, seq_len] 目标 token IDs (训练时)
            kv_caches: 每层的 KV 缓存列表 (推理时)
            position_offset: 位置偏移 (用于 KV cache 推理)

        Returns:
            (logits, loss, new_kv_caches)
        """
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len, \
            f"序列长度 {T} 超过最大长度 {self.config.max_seq_len}"

        # Token Embedding
        x = self.embed_dropout(self.token_embedding(input_ids))

        # 逐层通过 Transformer Blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(
                x,
                self.rope_cos_freqs,
                self.rope_sin_freqs,
                kv_cache=layer_kv_cache,
                position_offset=position_offset,
                use_flash=(kv_caches is None or all(kc is None for kc in kv_caches)),  # 有 KV cache 时不用 Flash Attention
            )
            new_kv_caches.append(new_kv)

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

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成 (使用 KV Cache 加速)

        Args:
            input_ids: [batch, seq_len] 初始 token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样参数
            top_p: Top-P (nucleus) 采样参数
            repetition_penalty: 重复惩罚系数
            eos_token_id: 结束 token ID (遇到则停止)

        Returns:
            生成的完整序列
        """
        self.eval()
        device = input_ids.device
        B = input_ids.size(0)

        # Prefill: 处理初始输入
        kv_caches = [None] * self.config.n_layers
        logits, _, kv_caches = self(input_ids, kv_caches=kv_caches)

        # 取最后一个 token 的 logits
        next_logits = logits[:, -1, :]

        generated = input_ids

        for step in range(max_new_tokens):
            # 温度缩放
            if temperature > 0:
                next_logits = next_logits / temperature

            # 重复惩罚
            if repetition_penalty != 1.0:
                for token_id in generated[0].tolist():
                    next_logits[0, token_id] /= repetition_penalty

            # Top-K 采样
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Top-P (Nucleus) 采样
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # 采样
            if temperature > 0:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # EOS 检查
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            # Decode: 用 KV Cache 逐 token 推理
            position_offset = generated.size(1) - 1
            logits, _, kv_caches = self(
                next_token,
                kv_caches=kv_caches,
                position_offset=position_offset,
            )
            next_logits = logits[:, -1, :]

        return generated

    def get_num_params(self, non_embedding: bool = True) -> int:
        """获取参数数量 (默认排除 embedding)"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params


# ──────────────── 模型保存/加载 ────────────────

def save_model(
    model: GPT,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    best_loss: float = float('inf'),
) -> None:
    """
    保存模型检查点

    Args:
        model: 模型实例
        path: 保存路径
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)
        step: 当前训练步数
        best_loss: 最佳损失
    """
    checkpoint = {
        "step": step,
        "best_loss": best_loss,
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"[保存] 检查点 -> {path} (step={step}, best_loss={best_loss:.4f})")


def load_model(
    path: str,
    device: torch.device = torch.device("cpu"),
    load_optimizer: bool = True,
) -> Tuple[GPT, Optional[dict], Optional[dict], int, float]:
    """
    加载模型检查点

    Args:
        path: 检查点路径
        device: 加载设备
        load_optimizer: 是否加载优化器状态

    Returns:
        (model, optimizer_state, scheduler_state, step, best_loss)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = GPTConfig.from_dict(checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    optimizer_state = checkpoint.get("optimizer_state_dict") if load_optimizer else None
    scheduler_state = checkpoint.get("scheduler_state_dict")
    step = checkpoint.get("step", 0)
    best_loss = checkpoint.get("best_loss", float('inf'))

    print(f"[加载] 检查点 <- {path} (step={step}, best_loss={best_loss:.4f})")
    return model, optimizer_state, scheduler_state, step, best_loss


# ──────────────── 主函数 (测试) ────────────────

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  FishAI v3 — 小体积最聪明的自研 Transformer")
    print(f"{'='*60}")

    for size in ["small", "medium", "large"]:
        config = get_model_config(size)
        print(f"\n--- {size.upper()} 配置 ---")
        print(f"  d_model:      {config.d_model}")
        print(f"  n_heads:      {config.n_heads} (Q)")
        print(f"  n_kv_heads:   {config.n_kv_heads} (KV, GQA)")
        print(f"  n_layers:     {config.n_layers}")
        print(f"  d_ff:         {config.d_ff} (SwiGLU)")
        print(f"  vocab_size:   {config.vocab_size}")
        print(f"  max_seq_len:  {config.max_seq_len}")
        print(f"  weight_tying: {config.weight_tying}")
        print(f"  总参数量:     {config.total_params() / 1e6:.1f}M")
        print(f"  量化后:       {config.quantized_size_mb():.1f} MB")

    # 测试前向传播
    print(f"\n--- 前向传播测试 (small) ---")
    config = get_model_config("small")
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss, _ = model(x, labels=x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # 测试生成 (KV Cache)
    print(f"\n--- 生成测试 (KV Cache) ---")
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(prompt, max_new_tokens=16, temperature=0.8)
    print(f"  Prompt: {prompt.shape}")
    print(f"  Generated: {generated.shape}")
