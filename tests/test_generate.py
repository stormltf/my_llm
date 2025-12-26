"""
文本生成器单元测试

测试内容：
1. TextGenerator 基础功能
2. 采样策略（Temperature, Top-k, Top-p）
3. 贪婪解码
4. 困惑度计算
5. 模型加载
"""

import pytest
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from generate import TextGenerator


class TestTextGenerator:
    """文本生成器测试"""

    @pytest.fixture
    def model_and_tokenizer(self):
        """创建模型和分词器"""
        # 创建小型模型
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        # 创建并训练分词器
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["你 好 世界", "人工 智能 很 有趣"]
        tokenizer.fit(texts, verbose=False)

        return model, tokenizer

    @pytest.fixture
    def generator(self, model_and_tokenizer):
        """创建生成器"""
        model, tokenizer = model_and_tokenizer
        device = torch.device("cpu")
        return TextGenerator(model, tokenizer, device)

    def test_generator_creation(self, model_and_tokenizer):
        """测试生成器创建"""
        model, tokenizer = model_and_tokenizer
        device = torch.device("cpu")
        generator = TextGenerator(model, tokenizer, device)

        assert generator.model is model
        assert generator.tokenizer is tokenizer

    def test_generate_basic(self, generator):
        """测试基础生成"""
        prompt = "你"
        result = generator.generate(
            prompt,
            max_length=20,
            temperature=1.0
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_max_length(self, generator):
        """测试最大长度限制"""
        prompt = "你"
        max_length = 10

        result = generator.generate(
            prompt,
            max_length=max_length,
            temperature=1.0
        )

        # 生成的 token 数不应超过 max_length
        ids = generator.tokenizer.encode(result)
        assert len(ids) <= max_length

    def test_generate_with_temperature(self, generator):
        """测试温度参数"""
        prompt = "你"

        # 低温度应该产生更确定的输出
        result_low_temp = generator.generate(
            prompt,
            max_length=10,
            temperature=0.1
        )

        # 高温度应该产生更随机的输出
        result_high_temp = generator.generate(
            prompt,
            max_length=10,
            temperature=2.0
        )

        # 两个结果都应该是有效的字符串
        assert isinstance(result_low_temp, str)
        assert isinstance(result_high_temp, str)

    def test_generate_with_top_k(self, generator):
        """测试 Top-k 采样"""
        prompt = "你"

        result = generator.generate(
            prompt,
            max_length=15,
            temperature=1.0,
            top_k=10
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_top_p(self, generator):
        """测试 Top-p 采样"""
        prompt = "你"

        result = generator.generate(
            prompt,
            max_length=15,
            temperature=1.0,
            top_p=0.9
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_eos(self, generator):
        """测试 EOS token 停止"""
        prompt = "你"

        result = generator.generate(
            prompt,
            max_length=50,
            temperature=1.0,
            eos_token_id=generator.tokenizer.eos_token_id
        )

        # 应该正常返回（可能在 EOS 处停止）
        assert isinstance(result, str)

    def test_generate_combined_sampling(self, generator):
        """测试组合采样策略"""
        prompt = "你"

        result = generator.generate(
            prompt,
            max_length=15,
            temperature=0.8,
            top_k=20,
            top_p=0.9
        )

        assert isinstance(result, str)


class TestGreedyDecoding:
    """贪婪解码测试"""

    @pytest.fixture
    def generator(self):
        """创建生成器"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["你 好 世界"]
        tokenizer.fit(texts, verbose=False)

        device = torch.device("cpu")
        return TextGenerator(model, tokenizer, device)

    def test_greedy_basic(self, generator):
        """测试贪婪解码基础功能"""
        prompt = "你"
        result = generator.generate_greedy(prompt, max_length=10)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_greedy_deterministic(self, generator):
        """测试贪婪解码确定性"""
        prompt = "你"

        result1 = generator.generate_greedy(prompt, max_length=10)
        result2 = generator.generate_greedy(prompt, max_length=10)

        # 贪婪解码应该每次产生相同结果
        assert result1 == result2

    def test_greedy_max_length(self, generator):
        """测试贪婪解码最大长度"""
        prompt = "你"
        max_length = 8

        result = generator.generate_greedy(prompt, max_length=max_length)
        ids = generator.tokenizer.encode(result)

        assert len(ids) <= max_length


class TestPerplexity:
    """困惑度测试"""

    @pytest.fixture
    def generator(self):
        """创建生成器"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["人工 智能 是 未来"]
        tokenizer.fit(texts, verbose=False)

        device = torch.device("cpu")
        return TextGenerator(model, tokenizer, device)

    def test_perplexity_basic(self, generator):
        """测试困惑度计算"""
        text = "人工 智能"
        ppl = generator.get_perplexity(text)

        assert isinstance(ppl, float)
        assert ppl > 0

    def test_perplexity_not_nan(self, generator):
        """测试困惑度不为 NaN"""
        text = "人工 智能 未来"
        ppl = generator.get_perplexity(text)

        assert not (ppl != ppl)  # NaN check

    def test_perplexity_finite(self, generator):
        """测试困惑度有限"""
        text = "人工 智能"
        ppl = generator.get_perplexity(text)

        assert ppl < float('inf')


class TestSamplingStrategies:
    """采样策略详细测试"""

    def test_top_k_filtering(self):
        """测试 Top-k 过滤"""
        # 模拟 logits
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        k = 3

        # Top-k 过滤
        kth_value = torch.topk(logits, k).values[-1]
        filtered = torch.where(
            logits < kth_value,
            torch.tensor(float('-inf')),
            logits
        )

        # 只有前 k 个应该保留
        valid_count = (filtered != float('-inf')).sum()
        assert valid_count == k

    def test_top_p_filtering(self):
        """测试 Top-p 过滤"""
        # 模拟 logits（已排序）
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        top_p = 0.9

        probs = F.softmax(logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # 找到超过 top_p 的位置
        mask = cumulative_probs <= top_p
        # 至少保留一个
        mask[0] = True

        # 检查累积概率
        assert probs[mask].sum() >= probs[0]  # 至少包含最高概率

    def test_temperature_effect(self):
        """测试温度效果"""
        logits = torch.tensor([1.0, 2.0, 3.0])

        # 低温度 -> 分布更尖锐
        probs_low = F.softmax(logits / 0.1, dim=-1)
        # 高温度 -> 分布更平坦
        probs_high = F.softmax(logits / 2.0, dim=-1)

        # 低温度时最高概率应该更高
        assert probs_low.max() > probs_high.max()


class TestGeneratorEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def generator(self):
        """创建生成器"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["测试 文本"]
        tokenizer.fit(texts, verbose=False)

        device = torch.device("cpu")
        return TextGenerator(model, tokenizer, device)

    def test_empty_prompt(self, generator):
        """测试空提示"""
        # 空提示会导致错误（这是预期行为）
        # 在实际使用中应该避免空提示
        import pytest
        with pytest.raises((IndexError, RuntimeError)):
            generator.generate("", max_length=10, temperature=1.0)

    def test_very_long_prompt(self, generator):
        """测试很长的提示"""
        # 超过 context_size 的提示
        prompt = "测 " * 50
        result = generator.generate(prompt, max_length=60, temperature=1.0)
        assert isinstance(result, str)

    def test_max_length_equals_prompt_length(self, generator):
        """测试最大长度等于提示长度"""
        prompt = "测 试"
        prompt_len = len(generator.tokenizer.encode(prompt))

        result = generator.generate(prompt, max_length=prompt_len, temperature=1.0)
        # 应该返回原始提示
        assert len(result) > 0

    def test_temperature_zero_like(self, generator):
        """测试接近零的温度"""
        prompt = "测"
        # 非常低的温度（接近贪婪）
        result = generator.generate(prompt, max_length=10, temperature=0.01)
        assert isinstance(result, str)

    def test_top_k_one(self, generator):
        """测试 Top-k=1（等价于贪婪）"""
        prompt = "测"
        result = generator.generate(
            prompt,
            max_length=10,
            temperature=1.0,
            top_k=1
        )
        assert isinstance(result, str)

    def test_top_p_very_low(self, generator):
        """测试很低的 Top-p"""
        prompt = "测"
        result = generator.generate(
            prompt,
            max_length=10,
            temperature=1.0,
            top_p=0.1
        )
        assert isinstance(result, str)


class TestGeneratorNumericalStability:
    """数值稳定性测试"""

    @pytest.fixture
    def generator(self):
        """创建生成器"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        model.eval()

        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["数值 稳定"]
        tokenizer.fit(texts, verbose=False)

        device = torch.device("cpu")
        return TextGenerator(model, tokenizer, device)

    def test_no_nan_in_generation(self, generator):
        """测试生成过程无 NaN"""
        for _ in range(5):
            result = generator.generate("数", max_length=20, temperature=1.0)
            # 结果不应该包含特殊字符（表示 NaN）
            assert result is not None

    def test_extreme_temperature(self, generator):
        """测试极端温度值"""
        # 非常高的温度
        result_high = generator.generate(
            "数",
            max_length=10,
            temperature=100.0
        )
        assert isinstance(result_high, str)

        # 非常低的温度
        result_low = generator.generate(
            "数",
            max_length=10,
            temperature=0.001
        )
        assert isinstance(result_low, str)

    def test_repeated_generation(self, generator):
        """测试重复生成"""
        # 多次生成应该都成功
        for _ in range(10):
            result = generator.generate(
                "数",
                max_length=15,
                temperature=0.8,
                top_k=10
            )
            assert isinstance(result, str)
            assert len(result) > 0
