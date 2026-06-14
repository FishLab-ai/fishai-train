"""
FishAI v3 数据处理工具

功能:
1. 下载和准备常见训练数据集 (HuggingFace datasets)
2. 数据质量过滤 (困惑度过滤、去重、语言检测)
3. 数据混合: 可配置比例 (web/code/books/wiki/medical)
4. 转换为分词后的二进制格式 (快速训练)
5. 数据集统计报告
"""

import os
import json
import hashlib
import argparse
import struct
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator, Tuple
from collections import Counter

import numpy as np


# ──────────────── 数据配置 ────────────────

DEFAULT_MIX_RATIOS = {
    "web": 0.50,       # 网页文本 (CommonCrawl 等)
    "code": 0.20,      # 代码 (The Stack, StarCoder)
    "books": 0.10,     # 书籍 (Books3)
    "wiki": 0.10,      # 维基百科
    "medical": 0.05,   # 医学文本
    "math": 0.05,      # 数学推理
}


# ──────────────── 数据加载 ────────────────

def load_text_file(path: str, encoding: str = "utf-8") -> str:
    """
    加载文本文件

    Args:
        path: 文件路径
        encoding: 编码格式

    Returns:
        文本内容
    """
    with open(path, "r", encoding=encoding, errors="replace") as f:
        return f.read()


def load_jsonl_file(path: str, text_key: str = "text") -> List[str]:
    """
    加载 JSONL 文件

    Args:
        path: JSONL 文件路径
        text_key: 文本字段名

    Returns:
        文本列表
    """
    texts = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    return texts


def load_documents(
    path: str,
    text_key: str = "text",
) -> List[str]:
    """
    智能加载文档 (自动识别文件格式)

    Args:
        path: 文件路径或目录路径
        text_key: JSONL 文本字段名

    Returns:
        文档列表
    """
    path_obj = Path(path)
    documents = []

    if path_obj.is_dir():
        # 递归加载目录
        for file_path in sorted(path_obj.rglob("*")):
            if file_path.suffix in (".txt", ".jsonl", ".json"):
                docs = load_documents(str(file_path), text_key)
                documents.extend(docs)
    elif path_obj.suffix == ".jsonl":
        documents = load_jsonl_file(path, text_key)
    elif path_obj.suffix == ".json":
        # JSON 数组格式
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    documents.append(item)
                elif isinstance(item, dict) and text_key in item:
                    documents.append(item[text_key])
    elif path_obj.suffix in (".txt", ".md", ".py", ".js", ".java", ".cpp", ".c", ".rs"):
        text = load_text_file(path)
        # 按双换行分段
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                documents.append(para)
    else:
        # 尝试作为纯文本加载
        try:
            text = load_text_file(path)
            for para in text.split("\n\n"):
                para = para.strip()
                if para:
                    documents.append(para)
        except Exception as e:
            print(f"[警告] 无法加载文件 {path}: {e}")

    return documents


# ──────────────── 数据质量过滤 ────────────────

class DataFilter:
    """
    数据质量过滤器

    支持:
    - 最小/最大长度过滤
    - 重复文档去重 (MinHash)
    - 语言检测 (简单启发式)
    - 困惑度过滤 (需要参考模型)
    - 特殊字符比例过滤
    """

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 100000,
        min_avg_word_length: float = 2.0,
        max_avg_word_length: float = 15.0,
        max_special_char_ratio: float = 0.3,
        max_digit_ratio: float = 0.5,
        deduplicate: bool = True,
        languages: Optional[List[str]] = None,
    ):
        """
        Args:
            min_length: 最小文档长度 (字符数)
            max_length: 最大文档长度
            min_avg_word_length: 最小平均词长
            max_avg_word_length: 最大平均词长
            max_special_char_ratio: 最大特殊字符比例
            max_digit_ratio: 最大数字比例
            deduplicate: 是否去重
            languages: 允许的语言代码 (None=全部允许)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_avg_word_length = min_avg_word_length
        self.max_avg_word_length = max_avg_word_length
        self.max_special_char_ratio = max_special_char_ratio
        self.max_digit_ratio = max_digit_ratio
        self.deduplicate = deduplicate
        self.languages = languages

        self._seen_hashes = set()
        self._stats = Counter()

    def _compute_hash(self, text: str) -> str:
        """计算文本哈希 (用于去重)"""
        # 归一化: 去除空白和标点差异
        normalized = "".join(c.lower() for c in text if c.isalnum())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def detect_language(self, text: str) -> str:
        """
        简单语言检测 (启发式)

        Returns:
            语言代码: "zh" (中文), "en" (英文), "code" (代码), "other"
        """
        # 统计字符类型
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return "other"

        chinese_ratio = chinese_chars / total_chars
        latin_ratio = latin_chars / total_chars

        # 代码检测
        code_keywords = ["def ", "class ", "import ", "function ", "var ",
                         "const ", "let ", "if (", "for (", "while ("]
        code_score = sum(1 for kw in code_keywords if kw in text)
        if code_score >= 2:
            return "code"

        if chinese_ratio > 0.3:
            return "zh"
        elif latin_ratio > 0.5:
            return "en"
        else:
            return "other"

    def filter_document(self, text: str) -> bool:
        """
        过滤单个文档

        Returns:
            True = 保留, False = 过滤掉
        """
        # 长度过滤
        if len(text) < self.min_length:
            self._stats["filtered_too_short"] += 1
            return False

        if len(text) > self.max_length:
            self._stats["filtered_too_long"] += 1
            return False

        # 去重
        if self.deduplicate:
            doc_hash = self._compute_hash(text)
            if doc_hash in self._seen_hashes:
                self._stats["filtered_duplicate"] += 1
                return False
            self._seen_hashes.add(doc_hash)

        # 平均词长
        words = text.split()
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            if avg_word_len < self.min_avg_word_length or avg_word_len > self.max_avg_word_length:
                self._stats["filtered_avg_word_length"] += 1
                return False

        # 特殊字符比例
        total_chars = len(text)
        if total_chars > 0:
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            special_ratio = special_chars / total_chars
            if special_ratio > self.max_special_char_ratio:
                self._stats["filtered_special_chars"] += 1
                return False

            # 数字比例
            digit_chars = sum(1 for c in text if c.isdigit())
            digit_ratio = digit_chars / total_chars
            if digit_ratio > self.max_digit_ratio:
                self._stats["filtered_digit_ratio"] += 1
                return False

        # 语言检测
        if self.languages is not None:
            lang = self.detect_language(text)
            if lang not in self.languages and lang != "other":
                self._stats["filtered_language"] += 1
                return False

        self._stats["kept"] += 1
        return True

    def filter_documents(self, documents: List[str]) -> List[str]:
        """批量过滤文档"""
        return [doc for doc in documents if self.filter_document(doc)]

    def get_stats(self) -> Dict[str, int]:
        """获取过滤统计"""
        return dict(self._stats)


# ──────────────── 数据混合 ────────────────

class DataMixer:
    """
    数据混合器
    按可配置比例混合不同来源的数据
    """

    def __init__(
        self,
        ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        """
        Args:
            ratios: 数据源比例 {"web": 0.5, "code": 0.2, ...}
            seed: 随机种子
        """
        self.ratios = ratios or DEFAULT_MIX_RATIOS.copy()
        self.rng = np.random.RandomState(seed)

        # 归一化比例
        total = sum(self.ratios.values())
        if total > 0:
            self.ratios = {k: v / total for k, v in self.ratios.items()}

        self._sources: Dict[str, List[str]] = {}

    def add_source(self, name: str, documents: List[str]) -> None:
        """添加数据源"""
        self._sources[name] = documents

    def mix(self, total_docs: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        按比例混合数据

        Args:
            total_docs: 总文档数 (None = 使用所有可用数据)

        Returns:
            [(source_name, document_text), ...]
        """
        if not self._sources:
            raise ValueError("未添加任何数据源")

        # 计算每个源应抽取的文档数
        if total_docs is None:
            total_docs = sum(len(docs) for docs in self._sources.values())

        source_counts = {}
        remaining = total_docs
        sources_list = list(self.ratios.keys())

        for i, source in enumerate(sources_list):
            if source not in self._sources:
                continue
            if i == len(sources_list) - 1:
                source_counts[source] = remaining
            else:
                count = int(total_docs * self.ratios.get(source, 0))
                # 不超过可用数据量
                count = min(count, len(self._sources[source]))
                source_counts[source] = count
                remaining -= count

        # 抽样
        mixed = []
        for source, count in source_counts.items():
            docs = self._sources[source]
            if count >= len(docs):
                sampled = docs
            else:
                indices = self.rng.choice(len(docs), size=count, replace=False)
                sampled = [docs[i] for i in indices]

            for doc in sampled:
                mixed.append((source, doc))

        # 打乱顺序
        self.rng.shuffle(mixed)

        return mixed


# ──────────────── 二进制格式转换 ────────────────

# 二进制格式说明:
# 头部:
#   magic: 4 bytes = "FAID" (FishAI Data)
#   version: 4 bytes = uint32
#   vocab_size: 4 bytes = uint32
#   n_docs: 4 bytes = uint32
#   max_seq_len: 4 bytes = uint32
# 文档索引:
#   每个文档: offset (8 bytes) + length (4 bytes) + source_id (4 bytes)
# 数据:
#   每个 token: uint32 (4 bytes)

MAGIC = b"FAID"
VERSION = 1
HEADER_SIZE = 20  # 4 + 4 + 4 + 4 + 4


def documents_to_binary(
    documents: List[Tuple[str, str]],
    tokenizer,
    output_path: str,
    max_seq_len: int = 2048,
    stride: int = 1024,
) -> Dict[str, Any]:
    """
    将文档转换为分词后的二进制格式 (用于快速训练)

    Args:
        documents: [(source_name, text), ...] 文档列表
        tokenizer: 分词器 (必须有 encode 方法)
        output_path: 输出文件路径
        max_seq_len: 最大序列长度
        stride: 滑动窗口步长

    Returns:
        统计信息字典
    """
    print(f"[数据] 转换为二进制格式: {len(documents)} 个文档")

    # 源名称 -> ID 映射
    source_names = sorted(set(src for src, _ in documents))
    source_to_id = {name: i for i, name in enumerate(source_names)}

    # 分词所有文档
    all_chunks = []  # [(source_id, token_ids), ...]
    total_tokens = 0
    total_chars = 0

    for source, text in documents:
        source_id = source_to_id[source]
        total_chars += len(text)

        # 编码
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        total_tokens += len(token_ids)

        # 滑动窗口切分
        if len(token_ids) <= max_seq_len + 1:
            if len(token_ids) > 1:
                all_chunks.append((source_id, token_ids))
        else:
            for start in range(0, len(token_ids) - max_seq_len, stride):
                chunk = token_ids[start:start + max_seq_len + 1]
                if len(chunk) > 1:
                    all_chunks.append((source_id, chunk))

    n_chunks = len(all_chunks)
    print(f"[数据]   总文档: {len(documents)}")
    print(f"[数据]   总字符: {total_chars:,}")
    print(f"[数据]   总 token: {total_tokens:,}")
    print(f"[数据]   训练片段: {n_chunks:,} (max_seq_len={max_seq_len}, stride={stride})")

    # 写入二进制文件
    with open(output_path, "wb") as f:
        # 1. 头部
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", tokenizer.vocab_size))
        f.write(struct.pack("<I", n_chunks))
        f.write(struct.pack("<I", max_seq_len))

        # 2. 源名称表
        f.write(struct.pack("<I", len(source_names)))
        for name in source_names:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

        # 3. 文档索引 (offset, length, source_id)
        index_offset = f.tell()
        # 先占位，后面回填
        index_size = n_chunks * 16  # 8 + 4 + 4
        f.write(b"\x00" * index_size)

        # 4. 数据
        data_start = f.tell()
        for i, (source_id, token_ids) in enumerate(all_chunks):
            offset = f.tell() - data_start
            length = len(token_ids)

            # 更新索引
            current_pos = f.tell()
            f.seek(index_offset + i * 16)
            f.write(struct.pack("<Q", offset))
            f.write(struct.pack("<I", length))
            f.write(struct.pack("<I", source_id))
            f.seek(current_pos)

            # 写入 token 数据
            for tid in token_ids:
                f.write(struct.pack("<I", tid))

    file_size = os.path.getsize(output_path)
    print(f"[数据]   文件大小: {file_size / (1024 * 1024):.1f} MB")

    return {
        "n_docs": len(documents),
        "n_chunks": n_chunks,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": max_seq_len,
        "file_size_mb": file_size / (1024 * 1024),
        "sources": source_names,
    }


def read_binary_dataset(
    path: str,
) -> Dict[str, Any]:
    """
    读取二进制数据集的元信息

    Returns:
        元信息字典
    """
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"无效的文件格式: {magic}")

        version = struct.unpack("<I", f.read(4))[0]
        vocab_size = struct.unpack("<I", f.read(4))[0]
        n_chunks = struct.unpack("<I", f.read(4))[0]
        max_seq_len = struct.unpack("<I", f.read(4))[0]

        # 源名称
        n_sources = struct.unpack("<I", f.read(4))[0]
        sources = []
        for _ in range(n_sources):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            sources.append(name)

    return {
        "version": version,
        "vocab_size": vocab_size,
        "n_chunks": n_chunks,
        "max_seq_len": max_seq_len,
        "sources": sources,
        "file_size_mb": os.path.getsize(path) / (1024 * 1024),
    }


# ──────────────── 数据集统计 ────────────────

def compute_dataset_stats(documents: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    计算数据集统计信息

    Args:
        documents: [(source_name, text), ...]

    Returns:
        统计信息字典
    """
    total_docs = len(documents)
    total_chars = sum(len(text) for _, text in documents)
    total_words = sum(len(text.split()) for _, text in documents)

    # 按来源统计
    source_stats = {}
    for source, text in documents:
        if source not in source_stats:
            source_stats[source] = {"count": 0, "chars": 0, "words": 0}
        source_stats[source]["count"] += 1
        source_stats[source]["chars"] += len(text)
        source_stats[source]["words"] += len(text.split())

    # 语言分布
    lang_counter = Counter()
    filter_ = DataFilter(deduplicate=False)
    for _, text in documents[:1000]:  # 采样检测
        lang = filter_.detect_language(text)
        lang_counter[lang] += 1

    return {
        "total_docs": total_docs,
        "total_chars": total_chars,
        "total_words": total_words,
        "avg_doc_length": total_chars / max(total_docs, 1),
        "source_stats": source_stats,
        "language_distribution": dict(lang_counter),
    }


def print_dataset_stats(stats: Dict[str, Any]) -> None:
    """打印数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"  数据集统计")
    print(f"{'='*60}")
    print(f"  总文档数:    {stats['total_docs']:,}")
    print(f"  总字符数:    {stats['total_chars']:,}")
    print(f"  总词数:      {stats['total_words']:,}")
    print(f"  平均文档长度: {stats['avg_doc_length']:.0f} 字符")

    print(f"\n  按来源分布:")
    for source, s in stats.get("source_stats", {}).items():
        pct = s["count"] / max(stats["total_docs"], 1) * 100
        print(f"    {source:10s}: {s['count']:>8,} docs ({pct:.1f}%), "
              f"{s['chars']:>12,} chars")

    if stats.get("language_distribution"):
        print(f"\n  语言分布 (采样):")
        total_sampled = sum(stats["language_distribution"].values())
        for lang, count in stats["language_distribution"].items():
            pct = count / max(total_sampled, 1) * 100
            print(f"    {lang:6s}: {pct:.1f}%")


# ──────────────── HuggingFace datasets 集成 ────────────────

def download_dataset(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
    cache_dir: str = "data/cache",
) -> List[str]:
    """
    从 HuggingFace 下载数据集

    Args:
        dataset_name: 数据集名称 (如 "wikimedia/wikipedia", "bigcode/the-stack")
        split: 数据分割
        text_column: 文本列名
        max_samples: 最大样本数
        cache_dir: 缓存目录

    Returns:
        文档列表
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets: pip install datasets")

    print(f"[数据] 下载: {dataset_name} (split={split})")
    dataset = load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    documents = []
    for item in dataset:
        text = item.get(text_column, "")
        if text:
            documents.append(text)

    print(f"[数据]   下载完成: {len(documents)} 个文档")
    return documents


# ──────────────── 命令行入口 ────────────────

def main():
    parser = argparse.ArgumentParser(description="FishAI v3 数据处理工具")
    subparsers = parser.add_subparsers(dest="command")

    # filter: 数据过滤
    filter_parser = subparsers.add_parser("filter", help="数据质量过滤")
    filter_parser.add_argument("--input", type=str, required=True, help="输入目录/文件")
    filter_parser.add_argument("--output", type=str, required=True, help="输出目录")
    filter_parser.add_argument("--min-length", type=int, default=50, help="最小文档长度")
    filter_parser.add_argument("--max-length", type=int, default=100000, help="最大文档长度")
    filter_parser.add_argument("--no-dedup", action="store_true", help="不去重")
    filter_parser.add_argument("--languages", type=str, nargs="+", default=None,
                               help="允许的语言 (zh en code)")

    # stats: 数据统计
    stats_parser = subparsers.add_parser("stats", help="数据集统计")
    stats_parser.add_argument("--input", type=str, required=True, help="输入目录/文件")

    # convert: 转换为二进制
    convert_parser = subparsers.add_parser("convert", help="转换为二进制格式")
    convert_parser.add_argument("--input", type=str, required=True, help="输入目录/文件")
    convert_parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    convert_parser.add_argument("--tokenizer", type=str, required=True, help="分词器路径")
    convert_parser.add_argument("--max-seq-len", type=int, default=2048, help="最大序列长度")
    convert_parser.add_argument("--stride", type=int, default=1024, help="滑动窗口步长")

    # download: 下载数据集
    download_parser = subparsers.add_parser("download", help="下载数据集")
    download_parser.add_argument("--dataset", type=str, required=True, help="HuggingFace 数据集名")
    download_parser.add_argument("--split", type=str, default="train", help="数据分割")
    download_parser.add_argument("--output", type=str, required=True, help="输出目录")
    download_parser.add_argument("--max-samples", type=int, default=None, help="最大样本数")
    download_parser.add_argument("--text-column", type=str, default="text", help="文本列名")

    args = parser.parse_args()

    if args.command == "filter":
        data_filter = DataFilter(
            min_length=args.min_length,
            max_length=args.max_length,
            deduplicate=not args.no_dedup,
            languages=args.languages,
        )
        documents = load_documents(args.input)
        print(f"[过滤] 加载 {len(documents)} 个文档")
        filtered = data_filter.filter_documents(documents)
        print(f"[过滤] 保留 {len(filtered)} 个文档")
        print(f"[过滤] 统计: {data_filter.get_stats()}")

        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, "filtered.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in filtered:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        print(f"[过滤] 已保存 -> {output_file}")

    elif args.command == "stats":
        documents = load_documents(args.input)
        labeled = [("unknown", doc) for doc in documents]
        stats = compute_dataset_stats(labeled)
        print_dataset_stats(stats)

    elif args.command == "convert":
        from tokenizer_train import FishAITokenizer
        tokenizer = FishAITokenizer(args.tokenizer)
        documents = load_documents(args.input)
        labeled = [("default", doc) for doc in documents]
        info = documents_to_binary(
            labeled, tokenizer, args.output,
            max_seq_len=args.max_seq_len,
            stride=args.stride,
        )
        print(f"[转换] 完成: {info}")

    elif args.command == "download":
        docs = download_dataset(
            args.dataset,
            split=args.split,
            text_column=args.text_column,
            max_samples=args.max_samples,
        )
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, "data.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        print(f"[下载] 已保存 -> {output_file}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
