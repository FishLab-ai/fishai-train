"""
TinyAI - 完全自研的 GPT 模型定义

从零实现每一个组件：
1. Token Embedding + 位置编码
2. Multi-Head Causal Self-Attention
3. Feed-Forward Network (GELU)
4. Pre-LayerNorm Transformer Block
5. GPT 模型组装

架构参数 (~10M 参数):
- d_model: 512
- n_heads: 8
- n_layers: 6
- d_ff: 2048
- vocab_size: 32000
- max_seq_len: 512
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """GPT 模型配置"""
    vocab_size: int = 32000
    max_seq_len: int = 512
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    # 初始化参数
    init_range: float = 0.02

    def total_params(self) -> int:
        """计算总参数量"""
        # Token Embedding
        tok_emb = self.vocab_size * self.d_model
        # Position Embedding
        pos_emb = self.max_seq_len * self.d_model

        # Per Layer
        # Attention: Q, K, V, O projections (4 * d_model^2)
        attn = 4 * self.d_model * self.d_model + 4 * self.d_model  # +biases
        # FFN: W1(d_model, d_ff) + W2(d_ff, d_model) + biases
        ffn = self.d_model * self.d_ff + self.d_ff  # W1 + b1
        ffn += self.d_ff * self.d_model + self.d_model  # W2 + b2
        # LayerNorm: 2 per layer, each with gamma + beta
        ln = 4 * self.d_model

        layer_params = attn + ffn + ln
        transformer_params = layer_params * self.n_layers

        # Final LayerNorm
        final_ln = 2 * self.d_model
        # LM Head (tied with token embedding or separate)
        lm_head = self.d_model * self.vocab_size

        return tok_emb + pos_emb + transformer_params + final_ln + lm_head

    def quantized_size_mb(self) -> float:
        """4-bit 量化后大小"""
        return self.total_params() * 0.5 / (1024 * 1024)


class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力机制

    实现:
    - Q, K, V 线性投影
    - Scaled Dot-Product Attention
    - 因果掩码 (下三角矩阵)
    - 多头并行计算
    - 输出投影 + Dropout
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V 投影 (合并为一个线性层提高效率)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)
        # 输出投影
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果掩码
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # QKV 投影
        qkv = self.qkv_proj(x)  # [B, T, 3*C]
        q, k, v = qkv.split(self.d_model, dim=2)

        # 重塑为多头形式
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, nh, T, hd]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, nh, T, T]

        # 应用因果掩码
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和
        out = attn @ v  # [B, nh, T, hd]

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class FeedForward(nn.Module):
    """
    前馈神经网络 (两层 MLP)

    结构: x -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> Dropout
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block

    结构:
    x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
    |___________________________________________|_____|
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + Attention + Residual
        x = x + self.attn(self.ln1(x))
        # Pre-Norm + FFN + Residual
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    GPT 模型 (Decoder-Only Transformer)

    完全自研实现，包含:
    - Token Embedding
    - Position Embedding (可学习)
    - N 个 Transformer Block
    - 最终 LayerNorm
    - LM Head (语言模型头)
    - 权重初始化
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        # Position Embedding
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        # Dropout
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # 最终 LayerNorm
        self.final_ln = nn.LayerNorm(config.d_model)

        # LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重初始化
        self.apply(self._init_weights)

        print(f"[GPT] 模型参数量: {self.total_params() / 1e6:.1f}M")
        print(f"[GPT] 4-bit 量化后: {config.quantized_size_mb():.1f} MB")

    def _init_weights(self, module):
        """权重初始化: Normal(0, 0.02)"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_range)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            input_ids: [B, T] 输入 token IDs
            labels: [B, T] 目标 token IDs (用于计算 loss)

        Returns:
            logits: [B, T, V] 预测的 logits
            loss: 标量 (如果提供 labels)
        """
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len, f"序列长度 {T} 超过最大 {self.config.max_seq_len}"

        # Position indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Embeddings
        tok_emb = self.token_embedding(input_ids)    # [B, T, d_model]
        pos_emb = self.position_embedding(pos)        # [1, T, d_model]
        x = self.embed_dropout(tok_emb + pos_emb)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.final_ln(x)

        # LM Head
        logits = self.lm_head(x)  # [B, T, vocab_size]

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
        """
        自回归生成文本

        Args:
            input_ids: [1, T] 输入 prompt
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样
            top_p: Nucleus 采样

        Returns:
            生成的 token 序列 [1, T + max_new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 截断到最大上下文长度
            idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len \
                else input_ids[:, -self.config.max_seq_len:]

            # 前向传播
            logits, _ = self(idx_cond)

            # 取最后一个位置
            logits = logits[:, -1, :] / temperature

            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Nucleus (Top-P) 采样
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

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    # 测试模型
    config = GPTConfig()
    model = GPTConfig()

    print(f"\n模型配置:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  总参数量: {config.total_params() / 1e6:.1f}M")
    print(f"  4-bit 量化: {config.quantized_size_mb():.1f} MB")

    # 测试前向传播
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits, loss = model(x, labels=x)
    print(f"\n前向传播测试:")
    print(f"  输入: {x.shape}")
    print(f"  输出: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
