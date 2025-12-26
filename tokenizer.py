"""
BPE (Byte Pair Encoding) 分词器实现

BPE 是目前最流行的子词分词算法，核心思想是：
1. 从字符级别开始，逐步合并最高频的相邻字符对
2. 反复迭代，直到达到预设的词表大小

作者：根据《Build a Large Language Model (From Scratch)》实现
"""

import os
import json
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """
    BPE 分词器

    功能：
    1. fit(): 从语料中学习合并规则，构建词表
    2. encode(): 将文本编码为 token ID 序列
    3. decode(): 将 token ID 序列解码回文本
    """

    def __init__(self, vocab_size: int = 1000, special_tokens: Optional[Dict[str, int]] = None):
        """
        初始化分词器

        Args:
            vocab_size: 目标词表大小
            special_tokens: 特殊token，如 {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
        """
        self.vocab_size = vocab_size
        self.vocab = {}  # token -> id 的映射
        self.inverse_vocab = {}  # id -> token 的反向映射
        self.merges = []  # 存储合并规则 [(pair1, merged_token1), (pair2, merged_token2), ...]
        # 注意：本实现主要针对中文，按空格分词后直接按字符处理
        # 英文分词可以后续扩展

        # 初始化特殊token
        self.special_tokens = special_tokens or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<EOS>": 2,
            "<BOS>": 3
        }

        # 将特殊token加入词表
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token

        # 下一个可用的普通token ID (从特殊token数量之后开始)
        self.next_token_id = len(self.special_tokens)

    def _get_word_pairs(self, word: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        统计单词中所有相邻token对的频率

        Args:
            word: 格式为 {"token": count} 的字典，如 {("h", "e", "l", "l", "o"): 2}

        Returns:
            相邻token对的频率字典，如 {("h", "e"): 2, ("e", "l"): 4, ...}
        """
        pairs = defaultdict(int)
        (first_token, remaining_count) = next(iter(word.items()))

        # 遍历token序列的每一对相邻token
        for token in list(first_token)[1:]:
            prev_token = token
            pairs[(prev_token, token)] += remaining_count

        return dict(pairs)

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """
        在词表中合并指定的token对

        Args:
            pair: 要合并的token对，如 ("e", "r")
            vocab: 当前词表

        Returns:
            合并后的新词表
        """
        new_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word_tuple in vocab:
            # 将单词tuple转换为字符串，替换后再转回tuple
            word = " ".join(word_tuple)
            word = word.replace(bigram, replacement)
            new_vocab[tuple(word.split(" "))] = vocab[word_tuple]

        return new_vocab

    def fit(self, texts: List[str], verbose: bool = True) -> 'BPETokenizer':
        """
        训练分词器，学习BPE合并规则

        这是BPE的核心算法：
        1. 预处理：将所有文本拆分为字符序列
        2. 初始化：创建初始词表（包含所有字符）
        3. 迭代：重复以下步骤直到达到词表大小
           a. 统计所有相邻字符对的频率
           b. 找到最高频的字符对
           c. 将该字符对合并为新token
           d. 更新词表

        Args:
            texts: 训练文本列表
            verbose: 是否打印训练进度

        Returns:
            self (支持链式调用)
        """
        if verbose:
            print(f"开始训练 BPE 分词器，目标词表大小: {self.vocab_size}")
            print(f"特殊token: {self.special_tokens}")

        # 第一步：预处理文本，构建初始词表
        # 格式：{('字', '符', '序', '列'): 频率}
        vocab = defaultdict(int)

        for text in texts:
            # 简单按字符拆分（中文）或按空格拆分（英文）
            for word in text.split():
                # 为每个字符加空格分隔，便于后续合并
                word_chars = tuple(" ".join(list(word)).split(" "))
                vocab[word_chars] += 1

        # 添加所有单字符到词表
        for word_tuple in vocab:
            for char in word_tuple:
                if char not in self.vocab:
                    self.vocab[char] = self.next_token_id
                    self.inverse_vocab[self.next_token_id] = char
                    self.next_token_id += 1

        if verbose:
            print(f"初始字符数: {len(self.vocab) - len(self.special_tokens)}")

        # 第二步：BPE 迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        if num_merges <= 0:
            if verbose:
                print(f"警告: 词表大小 {self.vocab_size} 小于等于已有字符数，无需合并")
            return self

        for i in range(num_merges):
            # 统计所有相邻token对的频率
            pairs = defaultdict(int)
            for word_tuple, count in vocab.items():
                for j in range(len(word_tuple) - 1):
                    pair = (word_tuple[j], word_tuple[j + 1])
                    pairs[pair] += count

            if not pairs:
                if verbose:
                    print(f"在第 {i} 步时已无可合并的字符对")
                break

            # 找到最高频的字符对
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]

            # 合并该字符对
            vocab = self._merge_vocab(best_pair, vocab)

            # 将合并后的新token加入词表
            new_token = "".join(best_pair)
            self.vocab[new_token] = self.next_token_id
            self.inverse_vocab[self.next_token_id] = new_token
            self.next_token_id += 1

            # 记录合并规则
            self.merges.append(best_pair)

            if verbose and (i + 1) % 100 == 0:
                print(f"训练进度: {i + 1}/{num_merges}, 最新合并: {best_pair}")

        if verbose:
            print(f"训练完成！最终词表大小: {len(self.vocab)}")

        return self

    def _apply_bpe(self, word: str) -> List[str]:
        """
        对单个单词应用BPE规则进行分词

        Args:
            word: 输入单词

        Returns:
            分词后的token列表
        """
        if len(word) == 1:
            return [word]

        # 初始化：将单词拆分为字符序列
        word_tokens = list(word)

        # 应用所有学到的合并规则
        # 注意：要按合并顺序应用，先合并的是更基础的组合
        while len(word_tokens) > 1:
            # 找到可以合并的且优先级最高的（先学到的优先级高）token对
            merge_found = False
            for i in range(len(word_tokens) - 1):
                pair = (word_tokens[i], word_tokens[i + 1])
                if pair in self.merges:
                    # 合并这两个token
                    new_token = "".join(pair)
                    word_tokens = word_tokens[:i] + [new_token] + word_tokens[i + 2:]
                    merge_found = True
                    break

            if not merge_found:
                # 没有可以合并的token对了
                break

        return word_tokens

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token ID序列

        Args:
            text: 输入文本

        Returns:
            token ID列表
        """
        if not text:
            return []

        token_ids = []

        # 按空格分词，然后对每个词应用BPE
        for word in text.split():
            # 应用BPE分词
            bpe_tokens = self._apply_bpe(word)

            # 将每个token转换为ID
            for token in bpe_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # 处理未知token：尝试拆分为字符
                    for char in token:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
                        else:
                            token_ids.append(self.special_tokens["<UNK>"])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        将token ID序列解码回文本

        Args:
            token_ids: token ID列表

        Returns:
            解码后的文本
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                # 跳过特殊token（除了可能需要的分隔符）
                if not token.startswith("<") and not token.endswith(">"):
                    tokens.append(token)

        return "".join(tokens)

    def save(self, filepath: str):
        """
        保存分词器到文件

        Args:
            filepath: 保存路径
        """
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"分词器已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'BPETokenizer':
        """
        从文件加载分词器

        Args:
            filepath: 文件路径

        Returns:
            加载的分词器实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"], special_tokens=data["special_tokens"])
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.next_token_id = len(tokenizer.vocab)

        return tokenizer

    @property
    def pad_token_id(self) -> int:
        return self.special_tokens["<PAD>"]

    @property
    def unk_token_id(self) -> int:
        return self.special_tokens["<UNK>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_tokens["<EOS>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_tokens["<BOS>"]


class MyLLMTokenizer(BPETokenizer):
    """
    MyLLM 分词器 - 基于 BPE 算法

    这是 BPETokenizer 的封装类，提供更友好的接口，
    并支持对话格式的特殊 token。

    使用示例：
    ---------
    >>> tokenizer = MyLLMTokenizer()
    >>> tokenizer.build_vocab("训练文本...", max_vocab_size=1000)
    >>> ids = tokenizer.encode("你好")
    >>> text = tokenizer.decode(ids)
    """

    def __init__(self, vocab_path: Optional[str] = None, vocab_size: int = 6400):
        """
        初始化分词器

        参数：
            vocab_path: 词表文件路径（如果提供，则加载已有词表）
            vocab_size: 目标词表大小（构建新词表时使用）
        """
        # 定义特殊 token（包括对话格式）
        special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<|im_start|>": 4,
            "<|im_end|>": 5,
        }

        super().__init__(vocab_size=vocab_size, special_tokens=special_tokens)

        if vocab_path and os.path.exists(vocab_path):
            self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path: str):
        """从 JSON 文件加载词表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.vocab = data.get('vocab', {})
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = [tuple(m) for m in data.get('merges', [])]
        self.next_token_id = len(self.vocab)

        print(f"词表已从 {vocab_path} 加载")
        print(f"词表大小: {len(self.vocab)}")
        if self.merges:
            print(f"BPE 合并规则: {len(self.merges)} 条")

    def build_vocab(self, text: str, max_vocab_size: int = 6400):
        """
        从文本构建词表（使用 BPE 算法）

        参数：
            text: 训练文本
            max_vocab_size: 最大词表大小
        """
        print("=" * 50)
        print("开始构建词表（BPE 算法）...")
        print("=" * 50)

        # 重新初始化父类，使用正确的 vocab_size
        BPETokenizer.__init__(self, vocab_size=max_vocab_size, special_tokens=self.special_tokens)

        # 将文本分割成"词"
        # BPE 会在词内部寻找可合并的字符对
        # fit() 方法会将每个词拆分为字符元组，如 "机器" -> ("机", "器")
        words = []
        current_word = []

        for char in text:
            if char in ' \n\t，。？！、：；""''（）【】':
                # 遇到分隔符，保存当前词
                if current_word:
                    # 不添加空格，保持词的完整性
                    words.append(''.join(current_word))
                    current_word = []
            elif char.strip():
                current_word.append(char)

        # 保存最后一个词
        if current_word:
            words.append(''.join(current_word))

        # 构造训练语料格式
        # 每个词用空格分隔，作为一个句子传入
        # fit() 会按空格分割，得到各个词，再将每个词拆分为字符
        corpus = [' '.join(words)]

        print(f"原始文本长度: {len(text)} 字符")
        print(f"分割成 {len(words)} 个词")
        if words:
            print(f"示例词: {words[:5]}")

        # 调用父类的 fit 方法进行 BPE 训练
        self.fit(corpus, verbose=True)

        print(f"\n词表构建完成！")
        print(f"最终词表大小: {len(self.vocab)}")
        print(f"BPE 合并规则: {len(self.merges)} 条")
        print("=" * 50)

    def save(self, filepath: str):
        """
        保存分词器到 JSON 文件

        保存内容包括：
        - vocab: token 到 ID 的映射
        - merges: BPE 合并规则
        - special_tokens: 特殊 token
        - vocab_size: 词表大小
        """
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "vocab_size": len(self.vocab)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"词表已保存到: {filepath}")

    def load(self, filepath: str):
        """加载词表"""
        self._load_vocab(filepath)

    def get_vocab_size(self) -> int:
        """返回当前词表大小"""
        return len(self.vocab)

    @property
    def im_start_token_id(self) -> int:
        return self.special_tokens.get("<|im_start|>", 4)

    @property
    def im_end_token_id(self) -> int:
        return self.special_tokens.get("<|im_end|>", 5)


def create_chinese_vocab(texts: List[str], vocab_size: int = 6400) -> MyLLMTokenizer:
    """
    从中文文本创建 BPE 分词器

    参数：
        texts: 训练文本列表
        vocab_size: 目标词表大小

    返回：
        训练好的分词器
    """
    tokenizer = MyLLMTokenizer(vocab_size=vocab_size)

    # 合并所有文本
    all_text = ' '.join(texts)

    # 构建词表
    tokenizer.build_vocab(all_text, max_vocab_size=vocab_size)

    return tokenizer


def demo_tokenizer():
    """
    分词器使用示例
    """
    print("=" * 50)
    print("BPE 分词器示例")
    print("=" * 50)

    # 准备训练语料（简单的中文句子）
    corpus = [
        "我 喜欢 学习 人工智能",
        "人工智能 是 未来 的 趋势",
        "我 想 学习 机器 学习",
        "深度 学习 是 机器 学习 的 分支",
        "自然 语言 处理 很 有趣",
        "我 喜欢 编写 代码",
        "代码 可以 改变 世界",
        "大模型 是 人工智能 的 重要 方向"
    ]

    # 创建并训练分词器
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.fit(corpus)

    print("\n" + "=" * 50)
    print("词表内容 (部分):")
    print("=" * 50)
    for i, (token, idx) in enumerate(list(tokenizer.vocab.items())[:20]):
        print(f"  {token:10s} -> {idx}")
    print(f"  ... (共 {len(tokenizer.vocab)} 个token)")

    # 测试编码
    print("\n" + "=" * 50)
    print("编码测试")
    print("=" * 50)

    test_text = "我 喜欢 学习 人工智能"
    token_ids = tokenizer.encode(test_text)
    print(f"原文: {test_text}")
    print(f"编码: {token_ids}")

    # 测试解码
    decoded_text = tokenizer.decode(token_ids)
    print(f"解码: {decoded_text}")

    # 保存分词器
    print("\n" + "=" * 50)
    tokenizer.save("/Users/bytedance/Downloads/go/src/github.com/my_llm/data/tokenizer.json")


if __name__ == "__main__":
    demo_tokenizer()
