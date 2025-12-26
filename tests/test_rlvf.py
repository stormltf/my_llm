"""
RLVF 训练器单元测试

测试内容：
1. MathVerifier - 数学验证器
2. LogicVerifier - 逻辑验证器
3. CompositeVerifier - 组合验证器
4. RLVFConfig 配置
5. RLVFTrainer 训练器
"""

import pytest
import torch

from rlvf import (
    MathVerifier,
    LogicVerifier,
    CompositeVerifier,
    RLVFConfig,
    RLVFTrainer,
    load_rlvf_data
)
from config import MyLLMConfig
from model import MyLLM


class TestMathVerifier:
    """数学验证器测试"""

    @pytest.fixture
    def verifier(self):
        return MathVerifier()

    def test_extract_numbers(self, verifier):
        """测试数字提取"""
        text = "答案是42，还有3.14和-100"
        numbers = verifier.extract_numbers(text)

        assert 42 in numbers
        assert 3.14 in numbers
        assert -100 in numbers

    def test_extract_numbers_from_empty(self, verifier):
        """测试从空文本提取数字"""
        numbers = verifier.extract_numbers("")
        assert numbers == []

    def test_extract_numbers_no_numbers(self, verifier):
        """测试没有数字的文本"""
        numbers = verifier.extract_numbers("没有数字的文本")
        assert numbers == []

    def test_verify_correct_answer(self, verifier):
        """测试正确答案验证"""
        response = "答案是42"
        expected = "42"

        reward = verifier.verify(response, expected)
        assert reward == 1.0

    def test_verify_incorrect_answer(self, verifier):
        """测试错误答案验证"""
        response = "答案是50"
        expected = "42"

        reward = verifier.verify(response, expected)
        assert reward == 0.0

    def test_verify_multiple_numbers(self, verifier):
        """测试多数字回答"""
        response = "经过计算，2+3=5，所以答案是5"
        expected = "5"

        reward = verifier.verify(response, expected)
        assert reward == 1.0

    def test_verify_negative_numbers(self, verifier):
        """测试负数"""
        response = "结果是-42"
        expected = "-42"

        reward = verifier.verify(response, expected)
        assert reward == 1.0

    def test_verify_float_numbers(self, verifier):
        """测试浮点数"""
        response = "大约是3.14"
        expected = "3.14"

        reward = verifier.verify(response, expected)
        assert reward == 1.0


class TestLogicVerifier:
    """逻辑验证器测试"""

    @pytest.fixture
    def verifier(self):
        return LogicVerifier()

    def test_verify_with_keywords(self, verifier):
        """测试关键词验证"""
        response = "是的，猫会呼吸"
        expected = "是"
        keywords = ["是", "会"]

        reward = verifier.verify(response, expected, keywords)
        assert reward == 1.0

    def test_verify_with_keywords_fail(self, verifier):
        """测试关键词验证失败"""
        # 使用不包含任何目标关键词的响应
        response = "这个问题无法回答"
        expected = "是"
        keywords = ["是", "正确"]

        reward = verifier.verify(response, expected, keywords)
        assert reward == 0.0

    def test_verify_number_match(self, verifier):
        """测试数字匹配"""
        response = "还剩9只羊"
        expected = "9"

        reward = verifier.verify(response, expected, None)
        assert reward == 1.0

    def test_verify_number_mismatch(self, verifier):
        """测试数字不匹配"""
        response = "还剩5只"
        expected = "9"

        reward = verifier.verify(response, expected, None)
        assert reward == 0.0

    def test_verify_substring_match(self, verifier):
        """测试子串匹配"""
        response = "答案确实是巴黎"
        expected = "巴黎"

        reward = verifier.verify(response, expected, None)
        assert reward == 1.0

    def test_verify_substring_fail(self, verifier):
        """测试子串匹配失败"""
        response = "答案是伦敦"
        expected = "巴黎"

        reward = verifier.verify(response, expected, None)
        assert reward == 0.0

    def test_case_insensitive(self, verifier):
        """测试大小写不敏感"""
        response = "YES, that's correct"
        expected = "yes"

        reward = verifier.verify(response, expected, None)
        assert reward == 1.0


class TestCompositeVerifier:
    """组合验证器测试"""

    @pytest.fixture
    def verifier(self):
        return CompositeVerifier()

    def test_verify_math_task(self, verifier):
        """测试数学任务验证"""
        task = {
            'type': 'math',
            'expected_answer': '42'
        }
        response = "结果是42"

        reward = verifier.verify(task, response)
        assert reward == 1.0

    def test_verify_logic_task(self, verifier):
        """测试逻辑任务验证"""
        task = {
            'type': 'logic',
            'expected_answer': '是',
            'keywords': ['是', '会']
        }
        response = "是的，可以"

        reward = verifier.verify(task, response)
        assert reward == 1.0

    def test_verify_default_task(self, verifier):
        """测试默认任务类型（默认为 math）"""
        # 默认任务类型是 math，需要数字匹配
        task = {
            'expected_answer': '42'
        }
        response = "答案是42"

        reward = verifier.verify(task, response)
        assert reward == 1.0


class TestRLVFConfig:
    """RLVF 配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = RLVFConfig()

        assert config.clip_ratio == 0.2
        assert config.kl_coef == 0.01
        assert config.correct_reward == 1.0
        assert config.incorrect_reward == -0.5
        assert config.max_grad_norm == 1.0
        assert config.learning_rate == 1e-5
        assert config.max_new_tokens == 32
        assert config.num_iterations == 50
        assert config.samples_per_task == 2
        assert config.temperature == 0.7

    def test_custom_config(self):
        """测试自定义配置"""
        config = RLVFConfig(
            correct_reward=2.0,
            incorrect_reward=-1.0,
            num_iterations=10
        )

        assert config.correct_reward == 2.0
        assert config.incorrect_reward == -1.0
        assert config.num_iterations == 10


class TestRLVFTrainer:
    """RLVF 训练器测试"""

    @pytest.fixture
    def mini_config(self):
        return MyLLMConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=64,  # 增加上下文长度以支持生成测试
            dropout=0.0,
        )

    @pytest.fixture
    def policy_model(self, mini_config):
        return MyLLM(mini_config)

    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def encode(self, text):
                return [ord(c) % 100 for c in text[:32]]

            def decode(self, ids):
                return ''.join([chr(i + 32) for i in ids])

        return MockTokenizer()

    @pytest.fixture
    def rlvf_config(self):
        return RLVFConfig(
            num_iterations=2,
            samples_per_task=1,
            max_new_tokens=16
        )

    @pytest.fixture
    def trainer(self, policy_model, mock_tokenizer, mini_config, rlvf_config):
        return RLVFTrainer(
            policy_model=policy_model,
            tokenizer=mock_tokenizer,
            config=mini_config,
            rlvf_config=rlvf_config
        )

    def test_trainer_creation(self, trainer):
        """测试训练器创建"""
        assert trainer is not None
        assert trainer.policy is not None
        assert trainer.ref_policy is not None
        assert trainer.verifier is not None

    def test_ref_policy_frozen(self, trainer):
        """测试参考模型被冻结"""
        for param in trainer.ref_policy.parameters():
            assert not param.requires_grad

    def test_verifier_exists(self, trainer):
        """测试验证器存在"""
        assert isinstance(trainer.verifier, CompositeVerifier)

    def test_optimizer_exists(self, trainer):
        """测试优化器"""
        assert trainer.optimizer is not None

    def test_format_prompt(self, trainer):
        """测试提示格式化"""
        task = {'prompt': '1+1等于几？'}
        formatted = trainer._format_prompt(task)

        assert '<|im_start|>user' in formatted
        assert '1+1等于几？' in formatted
        assert '<|im_start|>assistant' in formatted

    def test_generate_and_verify(self, trainer):
        """测试生成和验证"""
        task = {
            'prompt': '1+1=',
            'type': 'math',
            'expected_answer': '2'
        }

        response, reward, full_ids, prompt_len = trainer.generate_and_verify(task)

        # 检查返回值
        assert isinstance(response, str)
        assert isinstance(reward, float)
        assert full_ids.size(0) == 1
        assert prompt_len > 0

    def test_train_step(self, trainer):
        """测试单步训练"""
        tasks = [
            {'prompt': '1+1=', 'type': 'math', 'expected_answer': '2'},
            {'prompt': '2+2=', 'type': 'math', 'expected_answer': '4'},
        ]

        metrics = trainer.train_step(tasks)

        # 检查返回指标
        assert 'loss' in metrics
        assert 'avg_reward' in metrics
        assert 'accuracy' in metrics
        assert 'num_samples' in metrics

    def test_evaluate(self, trainer):
        """测试评估"""
        tasks = [
            {'prompt': '测试', 'type': 'math', 'expected_answer': '42'},
        ]

        results = trainer.evaluate(tasks)

        assert 'accuracy' in results
        assert 'total' in results
        assert 'correct' in results
        assert 'results' in results

    def test_save_and_load_model(self, trainer, tmp_path):
        """测试模型保存和加载"""
        save_path = tmp_path / "test_rlvf_model.pt"

        # 保存
        trainer.save_model(str(save_path))
        assert save_path.exists()

        # 加载
        trainer.load_model(str(save_path))


class TestVerifierEdgeCases:
    """验证器边界情况测试"""

    def test_empty_response(self):
        """测试空回答"""
        verifier = MathVerifier()
        reward = verifier.verify("", "42")
        assert reward == 0.0

    def test_empty_expected(self):
        """测试空期望答案"""
        verifier = MathVerifier()
        reward = verifier.verify("答案是42", "")
        assert reward == 0.0

    def test_very_long_numbers(self):
        """测试很长的数字"""
        verifier = MathVerifier()
        reward = verifier.verify("答案是10000000000", "10000000000")
        assert reward == 1.0

    def test_unicode_numbers(self):
        """测试 Unicode 数字"""
        verifier = LogicVerifier()
        # 测试全角数字
        reward = verifier.verify("答案是４２", "42")
        # 可能不匹配，但不应报错

    def test_mixed_content(self):
        """测试混合内容"""
        verifier = MathVerifier()
        response = "经过仔细计算，我认为答案是42，但也不排除其他可能"
        reward = verifier.verify(response, "42")
        assert reward == 1.0


class TestRLVFIntegration:
    """RLVF 集成测试"""

    def test_full_training_loop(self):
        """测试完整训练循环"""
        config = MyLLMConfig(
            vocab_size=100,
            emb_dim=16,
            num_heads=2,
            num_layers=1,
            context_size=64,  # 增加上下文长度
            dropout=0.0,
        )

        policy = MyLLM(config)

        class MockTokenizer:
            def encode(self, text):
                return [ord(c) % 100 for c in text[:16]]

            def decode(self, ids):
                return ''.join([chr(i + 32) for i in ids])

        rlvf_config = RLVFConfig(
            num_iterations=2,
            samples_per_task=1
        )

        trainer = RLVFTrainer(policy, MockTokenizer(), config, rlvf_config)

        tasks = [
            {'prompt': '1+1=', 'type': 'math', 'expected_answer': '2'},
            {'prompt': '2+2=', 'type': 'math', 'expected_answer': '4'},
        ]

        history = trainer.train(tasks, batch_size=2)

        # 检查历史记录
        assert 'iteration' in history
        assert 'loss' in history
        assert 'reward' in history
        assert 'accuracy' in history

        # 检查迭代次数
        assert len(history['iteration']) == 2

    def test_math_then_logic_tasks(self):
        """测试混合数学和逻辑任务"""
        config = MyLLMConfig(
            vocab_size=100,
            emb_dim=16,
            num_heads=2,
            num_layers=1,
            context_size=64,  # 增加上下文长度
            dropout=0.0,
        )

        policy = MyLLM(config)

        class MockTokenizer:
            def encode(self, text):
                return [ord(c) % 100 for c in text[:16]]

            def decode(self, ids):
                return ''.join([chr(i + 32) for i in ids])

        rlvf_config = RLVFConfig(
            num_iterations=1,
            samples_per_task=1
        )

        trainer = RLVFTrainer(policy, MockTokenizer(), config, rlvf_config)

        tasks = [
            {'prompt': '1+1=', 'type': 'math', 'expected_answer': '2'},
            {'prompt': '猫会呼吸吗', 'type': 'logic', 'expected_answer': '是', 'keywords': ['是', '会']},
        ]

        metrics = trainer.train_step(tasks)

        # 应该处理两种类型的任务
        assert metrics['num_samples'] >= 0
