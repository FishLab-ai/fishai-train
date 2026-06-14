"""
FishAI v3 BPE 分词器训练

使用 HuggingFace tokenizers 库从零训练 BPE 分词器:
- 从训练数据学习 BPE 合并规则
- 支持添加特殊 token (chat template: system, user, assistant)
- 导出为 HuggingFace 兼容格式
- 支持与现有词表合并

默认配置:
- 词表大小: 32000
- 模型: BPE (Byte-Pair Encoding)
- 预分词: 类 GPT-2 的字节级 BPE
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from tokenizers import (
    Tokenizer,
    models,
    pre_tokenizers,
    decoders,
    trainers,
    processors,
)
from tokenizers.models import BPE


# ──────────────── 默认特殊 Token ────────────────

DEFAULT_SPECIAL_TOKENS = [
    "<PAD>",       # 填充
    "<BOS>",       # 序列开始
    "<EOS>",       # 序列结束
    "<UNK>",       # 未知
    "<MASK>",      # 掩码 (MLM 预训练)
    "<|system|>",  # 系统消息
    "<|user|>",    # 用户消息
    "<|assistant| >",  # 助手消息
    "<|end|>",     # 消息结束
    "<|thought|>", # 思维链
]


# ──────────────── BPE 分词器训练 ────────────────

def train_bpe_tokenizer(
    data_files: Union[str, List[str]],
    vocab_size: int = 32000,
    output_dir: str = "tokenizer",
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
    added_tokens: Optional[List[str]] = None,
    limit: Optional[int] = None,
    num_processes: Optional[int] = None,
) -> Tokenizer:
    """
    从训练数据训练 BPE 分词器

    Args:
        data_files: 训练数据文件路径 (支持 txt/jsonl)
        vocab_size: 目标词表大小
        output_dir: 输出目录
        min_frequency: 最小合并频率
        special_tokens: 特殊 token 列表
        added_tokens: 额外添加的 token (不会被 BPE 分割)
        limit: 限制处理的行数 (用于快速测试)
        num_processes: 并行进程数

    Returns:
        训练好的 Tokenizer 实例
    """
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS.copy()

    if added_tokens is None:
        added_tokens = []

    os.makedirs(output_dir, exist_ok=True)

    # 处理输入文件列表
    if isinstance(data_files, str):
        data_files = [data_files]

    # 验证文件存在
    valid_files = []
    for f in data_files:
        if os.path.exists(f):
            valid_files.append(f)
        else:
            print(f"[警告] 文件不存在，跳过: {f}")

    if not valid_files:
        raise FileNotFoundError(f"未找到任何有效的训练数据文件: {data_files}")

    print(f"[分词器] 训练 BPE 分词器")
    print(f"[分词器]   词表大小: {vocab_size}")
    print(f"[分词器]   训练数据: {len(valid_files)} 个文件")
    print(f"[分词器]   最小频率: {min_frequency}")
    print(f"[分词器]   特殊 token: {len(special_tokens)} 个")

    # 创建 BPE 模型
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # 设置预分词器 (类 GPT-2 字节级)
    # 将文本按空白和标点分割，再将每个词转为字节级表示
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
    )

    # 设置解码器
    tokenizer.decoder = decoders.ByteLevel()

    # 设置后处理器 (可选)
    # tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # 准备训练器
    all_special_tokens = special_tokens + added_tokens

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=all_special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # 训练
    print(f"[分词器] 开始训练...")
    tokenizer.train(
        valid_files,
        trainer=trainer,
    )

    # 保存
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"[分词器] 已保存 -> {tokenizer_path}")

    # 打印统计信息
    vocab = tokenizer.get_vocab()
    print(f"[分词器] 实际词表大小: {len(vocab)}")

    # 测试编码
    test_texts = [
        "你好，世界！",
        "Hello, world!",
        "FishAI 是一个 AI 助手。",
        "def hello(): print('hi')",
        "1 + 1 = 2",
    ]

    print(f"\n[分词器] 编码测试:")
    for text in test_texts:
        encoding = tokenizer.encode(text)
        print(f"  '{text}' -> {len(encoding.tokens)} tokens")
        print(f"    IDs: {encoding.ids[:20]}{'...' if len(encoding.ids) > 20 else ''}")

    return tokenizer


# ──────────────── 从 HuggingFace 预训练分词器初始化 ────────────────

def init_from_pretrained(
    pretrained_name: str,
    output_dir: str = "tokenizer",
    extra_special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    """
    从 HuggingFace 预训练分词器初始化

    Args:
        pretrained_name: HuggingFace 模型名 (如 "gpt2", "meta-llama/Llama-2-7b-hf")
        output_dir: 输出目录
        extra_special_tokens: 额外添加的特殊 token

    Returns:
        初始化后的 Tokenizer 实例
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[分词器] 从预训练分词器初始化: {pretrained_name}")
    tokenizer = Tokenizer.from_pretrained(pretrained_name)

    # 添加额外特殊 token
    if extra_special_tokens:
        tokenizer.add_special_tokens(extra_special_tokens)
        print(f"[分词器] 添加 {len(extra_special_tokens)} 个特殊 token")

    # 保存
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"[分词器] 已保存 -> {tokenizer_path}")

    return tokenizer


# ──────────────── 合并词表 ────────────────

def merge_vocabularies(
    base_tokenizer_path: str,
    extra_data_files: Union[str, List[str]],
    output_dir: str = "tokenizer_merged",
    extra_vocab_size: int = 1000,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    在现有分词器基础上，用额外数据扩展词表

    Args:
        base_tokenizer_path: 基础分词器路径
        extra_data_files: 额外训练数据
        output_dir: 输出目录
        extra_vocab_size: 额外词表大小
        min_frequency: 最小合并频率

    Returns:
        扩展后的 Tokenizer
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[分词器] 合并词表: 基础={base_tokenizer_path}")
    tokenizer = Tokenizer.from_file(base_tokenizer_path)

    current_vocab_size = tokenizer.get_vocab_size()
    target_vocab_size = current_vocab_size + extra_vocab_size

    if isinstance(extra_data_files, str):
        extra_data_files = [extra_data_files]

    # 继续训练
    trainer = trainers.BpeTrainer(
        vocab_size=target_vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(extra_data_files, trainer=trainer)

    # 保存
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"[分词器] 合并后词表大小: {tokenizer.get_vocab_size()}")
    print(f"[分词器] 已保存 -> {tokenizer_path}")

    return tokenizer


# ──────────────── FishAI 分词器封装 ────────────────

class FishAITokenizer:
    """
    FishAI 分词器封装
    统一接口，兼容训练和推理

    支持:
    - BPE 编码/解码
    - 特殊 token 处理
    - Chat template 格式化
    - 从 HuggingFace tokenizer.json 加载
    """

    def __init__(self, tokenizer_path: Optional[str] = None):
        """
        初始化分词器

        Args:
            tokenizer_path: tokenizer.json 文件路径 (None 则使用默认 BPE)
        """
        if tokenizer_path and os.path.exists(tokenizer_path):
            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            print(f"[FishAITokenizer] 从文件加载: {tokenizer_path}")
        else:
            # 创建一个简单的字节级 BPE 分词器作为 fallback
            self._tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self._tokenizer.decoder = decoders.ByteLevel()
            print(f"[FishAITokenizer] 使用默认字节级 BPE 分词器")

        # 特殊 token ID 映射
        self._build_special_token_map()

        # 启用 padding
        self._tokenizer.enable_padding(
            pad_id=self.pad_id,
            pad_token=self.pad_token,
        )

    def _build_special_token_map(self) -> None:
        """构建特殊 token 映射"""
        vocab = self._tokenizer.get_vocab()
        self._special_ids = {}
        self._special_tokens = {}

        special_names = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>",
                         "<|system|>", "<|user|>", "<|assistant| >", "<|end|>", "<|thought|>"]

        for name in special_names:
            if name in vocab:
                self._special_ids[name] = vocab[name]
                self._special_tokens[vocab[name]] = name

    @property
    def vocab_size(self) -> int:
        """词表大小"""
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        """PAD token ID"""
        return self._special_ids.get("<PAD>", 0)

    @property
    def pad_token(self) -> str:
        """PAD token"""
        return "<PAD>"

    @property
    def bos_id(self) -> int:
        """BOS token ID"""
        return self._special_ids.get("<BOS>", 1)

    @property
    def bos_token(self) -> str:
        """BOS token"""
        return "<BOS>"

    @property
    def eos_id(self) -> int:
        """EOS token ID"""
        return self._special_ids.get("<EOS>", 2)

    @property
    def eos_token(self) -> str:
        """EOS token"""
        return "<EOS>"

    @property
    def unk_id(self) -> int:
        """UNK token ID"""
        return self._special_ids.get("<UNK>", 3)

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """
        编码文本为 token ID 列表

        Args:
            text: 输入文本
            add_bos: 是否添加 BOS token
            add_eos: 是否添加 EOS token

        Returns:
            token ID 列表
        """
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        解码 token ID 列表为文本

        Args:
            token_ids: token ID 列表
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            解码后的文本
        """
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """
        将聊天消息编码为 token ID 列表

        格式:
        <|system|>system_message<|end|>
        <|user|>user_message<|end|>
        <|assistant| >assistant_message<|end|>

        Args:
            messages: 消息列表 [{"role": "system/user/assistant", "content": "..."}]
            add_generation_prompt: 是否添加生成提示

        Returns:
            token ID 列表
        """
        ids = [self.bos_id]

        role_tokens = {
            "system": "<|system|>",
            "user": "<|user|>",
            "assistant": "<|assistant| >",
        }

        vocab = self._tokenizer.get_vocab()

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # 角色标记
            role_token = role_tokens.get(role, "<|user|>")
            if role_token in vocab:
                ids.append(vocab[role_token])

            # 内容
            content_ids = self._tokenizer.encode(content).ids
            ids.extend(content_ids)

            # 结束标记
            if "<|end|>" in vocab:
                ids.append(vocab["<|end|>"])

        # 添加生成提示
        if add_generation_prompt:
            asst_token = "<|assistant| >"
            if asst_token in vocab:
                ids.append(vocab[asst_token])

        return ids

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> Dict[str, List[List[int]]]:
        """
        批量编码

        Args:
            texts: 文本列表
            max_length: 最大长度 (截断/填充)
            add_bos: 是否添加 BOS
            add_eos: 是否添加 EOS

        Returns:
            {"input_ids": [...], "attention_mask": [...]}
        """
        if max_length is not None:
            self._tokenizer.enable_truncation(max_length=max_length)
            self._tokenizer.enable_padding(
                pad_id=self.pad_id,
                pad_token=self.pad_token,
                length=max_length,
            )

        encodings = self._tokenizer.encode_batch(texts)

        input_ids = []
        attention_mask = []

        for encoding in encodings:
            ids = list(encoding.ids)
            mask = list(encoding.attention_mask)

            if add_bos:
                ids = [self.bos_id] + ids
                mask = [1] + mask
            if add_eos:
                ids = ids + [self.eos_id]
                mask = mask + [1]

            input_ids.append(ids)
            attention_mask.append(mask)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save(self, path: str) -> None:
        """保存分词器"""
        self._tokenizer.save(path)

    @classmethod
    def from_file(cls, path: str) -> "FishAITokenizer":
        """从文件加载"""
        return cls(tokenizer_path=path)


# ──────────────── 命令行入口 ────────────────

def main():
    parser = argparse.ArgumentParser(description="FishAI v3 BPE 分词器训练")
    parser.add_argument(
        "--data", type=str, nargs="+", required=True,
        help="训练数据文件路径 (支持 txt/jsonl)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32000,
        help="词表大小 (默认: 32000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="tokenizer",
        help="输出目录 (默认: tokenizer)",
    )
    parser.add_argument(
        "--min-frequency", type=int, default=2,
        help="最小合并频率 (默认: 2)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="限制处理的行数 (用于快速测试)",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="从 HuggingFace 预训练分词器初始化 (如 gpt2)",
    )
    parser.add_argument(
        "--merge-from", type=str, default=None,
        help="在现有分词器基础上扩展词表",
    )

    args = parser.parse_args()

    if args.pretrained:
        tokenizer = init_from_pretrained(
            args.pretrained,
            output_dir=args.output_dir,
            extra_special_tokens=DEFAULT_SPECIAL_TOKENS,
        )
    elif args.merge_from:
        tokenizer = merge_vocabularies(
            args.merge_from,
            args.data,
            output_dir=args.output_dir,
        )
    else:
        tokenizer = train_bpe_tokenizer(
            data_files=args.data,
            vocab_size=args.vocab_size,
            output_dir=args.output_dir,
            min_frequency=args.min_frequency,
            limit=args.limit,
        )

    # 保存配置文件 (HuggingFace 格式)
    config_path = os.path.join(args.output_dir, "tokenizer_config.json")
    config = {
        "model_type": "bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "model_max_length": 2048,
        "special_tokens": DEFAULT_SPECIAL_TOKENS,
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "unk_token": "<UNK>",
        "pad_token": "<PAD>",
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[分词器] 配置已保存 -> {config_path}")

    print(f"\n[分词器] 训练完成!")
    print(f"[分词器] 词表大小: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
