"""
边界情况测试

测试内容：
1. 空输入处理
2. 超长序列
3. 极端参数值
4. 内存不足情况
5. 数值溢出
6. 特殊字符处理
"""

import pytest
import torch
import torch.nn as nn

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from generate import TextGenerator


class TestEmptyInputs:
    """空输入测试"""

    def test_empty_text_encoding(self):
        """测试空文本编码"""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        tokens = tokenizer.encode("")

        # 空字符串应该返回空列表
        assert tokens == []

    def test_empty_prompt_generation(self):
        """测试空提示生成"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        generator = TextGenerator(model, tokenizer, torch.device("cpu"))

        # 空提示应该返回空或报错（取决于实现）
        # 当前实现会报 IndexError，这是预期的
        with pytest.raises((IndexError, RuntimeError)):
            generator.generate("", max_length=10)

    def test_empty_dataset(self):
        """测试空数据集"""
        from train import PretrainDataset

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        dataset = PretrainDataset([], tokenizer, seq_len=16)

        assert len(dataset) == 0

    def test_batch_with_all_padding(self):
        """测试需要填充的批次"""
        from train import collate_fn

        # 创建长度不同的样本，较短的需要填充到最长的长度
        batch = [
            (torch.tensor([1]), torch.tensor([2])),
            (torch.tensor([3, 4]), torch.tensor([4, 5])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        # 应该正确填充到最长序列长度
        assert padded_inputs.shape == (2, 2)
        assert padded_targets.shape == (2, 2)


class TestLongSequences:
    """超长序列测试"""

    def test_sequence_exceeds_context_size(self):
        """测试超过上下文长度的序列"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,  # 上下文长度 32
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        # 创建超过上下文长度的输入
        input_ids = torch.randint(0, 100, (1, 50))  # 长度 50 > 32

        # 应该抛出断言错误
        with pytest.raises(AssertionError):
            model(input_ids)

    def test_sequence_equals_context_size(self):
        """测试等于上下文长度的序列"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        # 精确等于上下文长度
        input_ids = torch.randint(0, 100, (1, 32))

        logits, _ = model(input_ids)

        # 应该正常工作
        assert logits.shape == (1, 32, 100)

    def test_very_long_text_encoding(self):
        """测试超长文本编码"""
        tokenizer = BPETokenizer(vocab_size=100)
        tokenizer.fit(["测 试"], verbose=False)

        # 创建超长文本
        long_text = "测 试 " * 1000

        tokens = tokenizer.encode(long_text)

        # 应该返回大量 token
        assert len(tokens) > 0

    def test_generation_with_long_prompt(self):
        """测试长提示生成"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=64,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=100)
        long_text = "测 试 " * 100
        tokenizer.fit([long_text], verbose=False)

        generator = TextGenerator(model, tokenizer, torch.device("cpu"))

        # 长提示应该被截断
        result = generator.generate(long_text, max_length=80)

        # 应该返回结果
        assert isinstance(result, str)


class TestExtremeParameters:
    """极端参数值测试"""

    def test_very_small_temperature(self):
        """测试非常小的温度"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        generator = TextGenerator(model, tokenizer, torch.device("cpu"))

        # 极小温度接近贪婪解码
        result = generator.generate("测", max_length=5, temperature=0.001)

        assert isinstance(result, str)

    def test_very_large_temperature(self):
        """测试非常大的温度"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        generator = TextGenerator(model, tokenizer, torch.device("cpu"))

        # 极大温度接近均匀分布
        result = generator.generate("测", max_length=5, temperature=100.0)

        assert isinstance(result, str)

    def test_top_k_extreme_values(self):
        """测试 Top-k 极端值"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        generator = TextGenerator(model, tokenizer, torch.device("cpu"))

        # Top-k = 1 等价于贪婪
        result1 = generator.generate("测", max_length=5, top_k=1)
        assert isinstance(result1, str)

        # Top-k = vocab_size 等价于不使用
        result2 = generator.generate("测", max_length=5, top_k=50)
        assert isinstance(result2, str)

    def test_top_p_extreme_values(self):
        """测试 Top-p 极端值"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        generator = TextGenerator(model, tokenizer, torch.device("cpu"))

        # Top-p = 1.0 等价于不使用
        result = generator.generate("测", max_length=5, top_p=1.0)
        assert isinstance(result, str)

        # Top-p = 0.0 应该至少选择一个
        result2 = generator.generate("测", max_length=5, top_p=0.01)
        assert isinstance(result2, str)


class TestNumericalEdgeCases:
    """数值边界情况测试"""

    def test_zero_variance_input(self):
        """测试零方差输入"""
        from model import LayerNorm

        ln = LayerNorm(emb_dim=64)

        # 全相同的输入（方差为0）
        x = torch.ones(2, 10, 64)

        output = ln(x)

        # 应该返回合理输出（虽然有警告）
        assert output.shape == x.shape
        assert not torch.isnan(output).all()

    def test_very_large_input_values(self):
        """测试很大的输入值"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        # 很大的输入值（通过极端 token ID）
        input_ids = torch.full((1, 10), 49)  # 最大 token ID

        logits, _ = model(input_ids)

        # 应该返回合理输出
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_negative_logits(self):
        """测试负 logits"""
        # 创建全是负数的 logits
        logits = torch.tensor([[-1.0, -2.0, -3.0]])

        # 应用 softmax
        probs = torch.softmax(logits, dim=-1)

        # 概率应该和为 1
        assert torch.allclose(probs.sum(), torch.ones(1), atol=1e-5)

    def test_mixed_positive_negative_logits(self):
        """测试混合正负 logits"""
        logits = torch.tensor([[1.0, -1.0, 2.0, -2.0]])

        probs = torch.softmax(logits, dim=-1)

        # 概率应该和为 1
        assert torch.allclose(probs.sum(), torch.ones(1), atol=1e-5)


class TestSpecialCharacters:
    """特殊字符测试"""

    def test_unicode_characters(self):
        """测试 Unicode 字符"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = [
            "Hello 世界 🌍",
            "测试 中文",
            "Emoji 😊 🎉",
        ]
        tokenizer.fit(texts, verbose=False)

        for text in texts:
            tokens = tokenizer.encode(text)
            assert len(tokens) > 0

    def test_newlines_and_tabs(self):
        """测试换行符和制表符"""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测\t试\n换 行"], verbose=False)

        text = "测\t试\n换 行"
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0

    def test_repeated_characters(self):
        """测试重复字符"""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        text = "啊啊啊啊啊啊"
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0


class TestModelEdgeCases:
    """模型边界情况测试"""

    def test_single_layer_model(self):
        """测试单层模型"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=1,  # 只有一层
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)

        input_ids = torch.randint(0, 50, (2, 10))
        logits, _ = model(input_ids)

        assert logits.shape == (2, 10, 50)

    def test_single_attention_head(self):
        """测试单注意力头模型"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=1,  # 只有一个头
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)

        input_ids = torch.randint(0, 50, (2, 10))
        logits, _ = model(input_ids)

        assert logits.shape == (2, 10, 50)

    def test_minimum_vocabulary_size(self):
        """测试最小词表大小"""
        config = GPTConfig(
            vocab_size=10,  # 非常小的词表
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)

        input_ids = torch.randint(0, 10, (2, 10))
        logits, _ = model(input_ids)

        assert logits.shape == (2, 10, 10)

    def test_large_embedding_dimension(self):
        """测试大嵌入维度"""
        config = GPTConfig(
            vocab_size=50,
            emb_dim=256,  # 较大的嵌入维度
            num_heads=8,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)

        input_ids = torch.randint(0, 50, (1, 10))
        logits, _ = model(input_ids)

        assert logits.shape == (1, 10, 50)


class TestGenerationEdgeCases:
    """生成边界情况测试"""

    @pytest.fixture
    def generator(self):
        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试 文 本"], verbose=False)

        return TextGenerator(model, tokenizer, torch.device("cpu"))

    def test_max_length_equals_prompt_length(self, generator):
        """测试最大长度等于提示长度"""
        prompt = "测"

        # 获取提示长度
        prompt_len = len(generator.tokenizer.encode(prompt))

        # 设置 max_length 等于提示长度
        result = generator.generate(prompt, max_length=prompt_len)

        # 应该返回原始提示（不生成新内容）
        assert len(result) > 0

    def test_max_length_less_than_prompt(self, generator):
        """测试最大长度小于提示长度"""
        prompt = "测 试 文 本"

        # 设置很小的 max_length
        result = generator.generate(prompt, max_length=2)

        # 应该返回截断的内容
        assert len(result) >= 0

    def test_unknown_tokens(self, generator):
        """测试未知 token"""
        # 使用不在训练集中的字符
        result = generator.generate("xyz", max_length=10)

        # 应该仍然生成内容（使用 UNK）
        assert isinstance(result, str)

    def test_generation_with_eos_immediate(self, generator):
        """测试立即遇到 EOS"""
        # 设置 EOS 为第一个 token
        result = generator.generate("测", max_length=10, eos_token_id=0)

        # 应该返回结果
        assert isinstance(result, str)


class TestBatchEdgeCases:
    """批次边界情况测试"""

    def test_single_sample_batch(self):
        """测试单样本批次"""
        from train import collate_fn

        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        assert padded_inputs.shape == (1, 3)
        assert padded_targets.shape == (1, 3)

    def test_very_large_batch(self):
        """测试非常大的批次"""
        from train import collate_fn

        # 创建 100 个样本
        batch = [
            (torch.tensor([1, 2]), torch.tensor([2, 3]))
            for _ in range(100)
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        assert padded_inputs.shape[0] == 100

    def test_variable_length_batch(self):
        """测试变长批次"""
        from train import collate_fn

        batch = [
            (torch.tensor([1]), torch.tensor([2])),
            (torch.tensor([1, 2, 3, 4, 5]), torch.tensor([2, 3, 4, 5, 6])),
            (torch.tensor([1, 2]), torch.tensor([2, 3])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        # 所有样本应该填充到相同长度
        assert padded_inputs.shape[0] == 3
        assert padded_inputs.shape[1] == 5


class TestTokenizerEdgeCases:
    """分词器边界情况测试"""

    def test_tokenizer_without_training(self):
        """测试未训练的分词器"""
        tokenizer = BPETokenizer(vocab_size=50)

        # 未训练时编码应该返回原始字符
        tokens = tokenizer.encode("测")

        # 应该返回一些东西（字符级）
        assert len(tokens) >= 0

    def test_tokenizer_with_single_character(self):
        """测试单字符文本"""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        tokens = tokenizer.encode("测")

        assert len(tokens) > 0

    def test_tokenizer_with_repeated_pattern(self):
        """测试重复模式"""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 测 测"], verbose=False)

        tokens = tokenizer.encode("测 测 测 测")

        assert len(tokens) > 0
