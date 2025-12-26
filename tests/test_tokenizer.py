"""
分词器单元测试

测试内容：
1. BPE 分词器基础功能
2. 编码/解码正确性
3. 特殊 token 处理
4. 保存/加载功能
5. MyLLMTokenizer 功能
"""

import os
import tempfile
import pytest
import torch

from tokenizer import BPETokenizer, MyLLMTokenizer, create_chinese_vocab


class TestBPETokenizer:
    """BPE 分词器测试"""

    def test_init_default(self):
        """测试默认初始化"""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 2000
        assert len(tokenizer.special_tokens) == 4
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1

    def test_init_custom_vocab_size(self):
        """测试自定义词表大小"""
        tokenizer = BPETokenizer(vocab_size=500)
        assert tokenizer.vocab_size == 500

    def test_init_custom_special_tokens(self):
        """测试自定义特殊 token"""
        special_tokens = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2}
        tokenizer = BPETokenizer(special_tokens=special_tokens)
        assert len(tokenizer.special_tokens) == 3
        assert "<SEP>" in tokenizer.vocab

    def test_fit_basic(self, sample_texts):
        """测试基础训练"""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(sample_texts, verbose=False)

        # 检查词表不为空
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
        # 检查所有单字符都在词表中
        for text in sample_texts:
            for char in text.replace(" ", ""):
                assert char in tokenizer.vocab

    def test_fit_merges(self, sample_texts):
        """测试 BPE 合并规则"""
        tokenizer = BPETokenizer(vocab_size=80)
        tokenizer.fit(sample_texts, verbose=False)

        # 检查有合并规则
        assert len(tokenizer.merges) > 0
        # 检查合并规则格式正确
        for merge in tokenizer.merges:
            assert isinstance(merge, tuple)
            assert len(merge) == 2

    def test_encode_basic(self, trained_tokenizer):
        """测试基础编码"""
        text = "人工智能"
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_string(self, trained_tokenizer):
        """测试空字符串编码"""
        ids = trained_tokenizer.encode("")
        assert ids == []

    def test_encode_return_tensors(self, trained_tokenizer):
        """测试返回 PyTorch tensor"""
        text = "学习"
        ids = trained_tokenizer.encode(text, return_tensors="pt")

        assert isinstance(ids, torch.Tensor)
        assert ids.dim() == 2
        assert ids.shape[0] == 1

    def test_decode_basic(self, trained_tokenizer):
        """测试基础解码"""
        original = "学习"
        ids = trained_tokenizer.encode(original)
        decoded = trained_tokenizer.decode(ids)

        # 解码结果应包含原始字符
        for char in original:
            assert char in decoded

    def test_encode_decode_roundtrip(self, trained_tokenizer):
        """测试编码-解码往返"""
        texts = ["人工智能", "机器学习", "深度学习"]
        for text in texts:
            ids = trained_tokenizer.encode(text)
            decoded = trained_tokenizer.decode(ids)
            # 解码结果去掉空格应等于原文
            assert decoded.replace(" ", "") == text.replace(" ", "")

    def test_decode_with_special_tokens(self, trained_tokenizer):
        """测试解码时跳过特殊 token"""
        # 手动构造包含特殊 token 的序列
        ids = [trained_tokenizer.pad_token_id, trained_tokenizer.unk_token_id]
        decoded = trained_tokenizer.decode(ids)
        # 特殊 token 应该被跳过
        assert "<PAD>" not in decoded
        assert "<UNK>" not in decoded

    def test_save_and_load(self, trained_tokenizer):
        """测试保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tokenizer.json")

            # 保存
            trained_tokenizer.save(filepath)
            assert os.path.exists(filepath)

            # 加载
            loaded = BPETokenizer.load(filepath)

            # 验证
            assert loaded.vocab_size == trained_tokenizer.vocab_size
            assert len(loaded.vocab) == len(trained_tokenizer.vocab)
            assert len(loaded.merges) == len(trained_tokenizer.merges)

    def test_vocab_consistency_after_load(self, trained_tokenizer):
        """测试加载后词表一致性"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tokenizer.json")
            trained_tokenizer.save(filepath)
            loaded = BPETokenizer.load(filepath)

            # 同一文本应产生相同的编码
            text = "人工智能"
            ids1 = trained_tokenizer.encode(text)
            ids2 = loaded.encode(text)
            assert ids1 == ids2


class TestMyLLMTokenizer:
    """MyLLM 分词器测试"""

    def test_init_default(self):
        """测试默认初始化"""
        tokenizer = MyLLMTokenizer()
        assert tokenizer.vocab_size == 2000
        assert tokenizer.im_start_token_id == 4
        assert tokenizer.im_end_token_id == 5

    def test_special_tokens_for_chat(self):
        """测试对话格式特殊 token"""
        tokenizer = MyLLMTokenizer()
        assert "<|im_start|>" in tokenizer.special_tokens
        assert "<|im_end|>" in tokenizer.special_tokens

    def test_build_vocab(self):
        """测试词表构建"""
        tokenizer = MyLLMTokenizer(vocab_size=100)
        text = "这是一段测试文本，用于构建词表。机器学习是人工智能的重要分支。"
        tokenizer.build_vocab(text, max_vocab_size=100)

        assert tokenizer.get_vocab_size() > 0

    def test_save_and_load(self):
        """测试保存和加载"""
        tokenizer = MyLLMTokenizer(vocab_size=50)
        text = "测试文本 用于 构建词表"
        tokenizer.build_vocab(text, max_vocab_size=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "vocab.json")
            tokenizer.save(filepath)

            # 创建新实例并加载
            new_tokenizer = MyLLMTokenizer(vocab_path=filepath)
            assert new_tokenizer.get_vocab_size() == tokenizer.get_vocab_size()


class TestCreateChineseVocab:
    """测试中文词表创建函数"""

    def test_create_vocab(self):
        """测试创建中文词表"""
        texts = ["你好世界", "人工智能", "机器学习"]
        tokenizer = create_chinese_vocab(texts, vocab_size=50)

        assert isinstance(tokenizer, MyLLMTokenizer)
        assert tokenizer.get_vocab_size() > 0


class TestTokenizerEdgeCases:
    """边界情况测试"""

    def test_unknown_character(self, trained_tokenizer):
        """测试未知字符处理"""
        # 使用一个不太可能在训练数据中的字符
        text = "你 好 世界"  # 假设"世界"不在训练数据中
        ids = trained_tokenizer.encode(text)
        # 不应该崩溃
        assert isinstance(ids, list)

    def test_very_long_text(self, trained_tokenizer):
        """测试很长的文本"""
        text = "人工智能 " * 100
        ids = trained_tokenizer.encode(text)
        assert len(ids) > 0

    def test_single_character(self, trained_tokenizer):
        """测试单字符文本"""
        text = "我"
        ids = trained_tokenizer.encode(text)
        assert len(ids) >= 1

    def test_repeated_fit(self, sample_texts):
        """测试重复训练"""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(sample_texts, verbose=False)
        vocab_size_1 = len(tokenizer.vocab)

        # 再次训练（应该重置）
        tokenizer.fit(sample_texts, verbose=False)
        vocab_size_2 = len(tokenizer.vocab)

        # 词表大小应该相同
        assert vocab_size_1 == vocab_size_2


class TestTokenizerProperties:
    """测试分词器属性"""

    def test_pad_token_id(self):
        """测试 PAD token ID"""
        tokenizer = BPETokenizer()
        assert tokenizer.pad_token_id == 0

    def test_unk_token_id(self):
        """测试 UNK token ID"""
        tokenizer = BPETokenizer()
        assert tokenizer.unk_token_id == 1

    def test_eos_token_id(self):
        """测试 EOS token ID"""
        tokenizer = BPETokenizer()
        # 根据代码，EOS token ID 应该是 special_tokens 中的值
        assert hasattr(tokenizer, 'eos_token_id')

    def test_bos_token_id(self):
        """测试 BOS token ID"""
        tokenizer = BPETokenizer()
        assert hasattr(tokenizer, 'bos_token_id')

    def test_case_insensitive_special_tokens(self):
        """测试特殊 token 大小写不敏感"""
        # 测试大写
        tokenizer1 = BPETokenizer(special_tokens={"<PAD>": 0, "<UNK>": 1})
        assert tokenizer1.pad_token_id == 0

        # 测试小写
        tokenizer2 = BPETokenizer(special_tokens={"<pad>": 0, "<unk>": 1})
        assert tokenizer2.pad_token_id == 0
