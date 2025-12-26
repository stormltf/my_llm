"""
配置单元测试

测试内容：
1. MyLLMConfig 基础功能
2. 配置验证
3. 保存/加载
4. 预设配置
5. RLHF/RLVF 配置
"""

import os
import tempfile
import pytest

from config import (
    MyLLMConfig, RLHFTrainConfig, RLVFTrainConfig,
    get_mini_config, get_small_config
)


class TestMyLLMConfig:
    """MyLLM 配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = MyLLMConfig()

        assert config.model_name == "my_llm"
        assert config.vocab_size == 2000
        assert config.emb_dim == 256
        assert config.num_heads == 4
        assert config.num_layers == 4
        assert config.context_size == 256
        assert config.dropout == 0.1

    def test_custom_values(self):
        """测试自定义值"""
        config = MyLLMConfig(
            model_name="custom_model",
            vocab_size=10000,
            emb_dim=512,
            num_heads=8
        )

        assert config.model_name == "custom_model"
        assert config.vocab_size == 10000
        assert config.emb_dim == 512
        assert config.num_heads == 8

    def test_head_dim_property(self):
        """测试 head_dim 属性"""
        config = MyLLMConfig(emb_dim=256, num_heads=4)
        assert config.head_dim == 64

        config2 = MyLLMConfig(emb_dim=512, num_heads=8)
        assert config2.head_dim == 64

    def test_validation_emb_dim_divisible(self):
        """测试 emb_dim 必须能被 num_heads 整除"""
        with pytest.raises(AssertionError):
            MyLLMConfig(emb_dim=256, num_heads=7)

    def test_special_token_ids(self):
        """测试特殊 token ID"""
        config = MyLLMConfig()

        assert config.pad_token_id == 0
        assert config.unk_token_id == 1
        assert config.bos_token_id == 2
        assert config.eos_token_id == 3
        assert config.im_start_token_id == 4
        assert config.im_end_token_id == 5

    def test_save_and_load(self):
        """测试保存和加载"""
        config = MyLLMConfig(
            model_name="test_model",
            vocab_size=1000,
            emb_dim=128,
            num_heads=4
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "config.json")

            # 保存
            config.save(filepath)
            assert os.path.exists(filepath)

            # 加载
            loaded = MyLLMConfig.load(filepath)

            assert loaded.model_name == config.model_name
            assert loaded.vocab_size == config.vocab_size
            assert loaded.emb_dim == config.emb_dim
            assert loaded.num_heads == config.num_heads

    def test_load_ignores_head_dim(self):
        """测试加载时忽略 head_dim（因为它是 property）"""
        config = MyLLMConfig(emb_dim=256, num_heads=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "config.json")
            config.save(filepath)

            # 手动添加 head_dim 到文件（模拟旧版本）
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            data['head_dim'] = 64
            with open(filepath, 'w') as f:
                json.dump(data, f)

            # 加载应该不报错
            loaded = MyLLMConfig.load(filepath)
            assert loaded.head_dim == 64

    def test_str_representation(self):
        """测试字符串表示"""
        config = MyLLMConfig()
        s = str(config)

        assert "MyLLM" in s
        assert str(config.vocab_size) in s
        assert str(config.emb_dim) in s

    def test_estimate_params(self):
        """测试参数量估算"""
        config = MyLLMConfig()
        params = config._estimate_params()

        assert isinstance(params, int)
        assert params > 0
        # 对于默认配置，应该约 3-5M 参数
        assert 1_000_000 < params < 10_000_000


class TestPresetConfigs:
    """预设配置测试"""

    def test_mini_config(self):
        """测试迷你配置"""
        config = get_mini_config()

        assert config.model_name == "my_llm-mini"
        assert config.vocab_size == 2000
        assert config.emb_dim == 256
        assert config.num_heads == 4
        assert config.num_layers == 4

    def test_small_config(self):
        """测试小型配置"""
        config = get_small_config()

        assert config.model_name == "my_llm-small"
        assert config.vocab_size == 2000
        assert config.emb_dim == 512
        assert config.num_heads == 8
        assert config.num_layers == 8

    def test_mini_smaller_than_small(self):
        """测试 mini 配置比 small 配置小"""
        mini = get_mini_config()
        small = get_small_config()

        assert mini._estimate_params() < small._estimate_params()


class TestRLHFTrainConfig:
    """RLHF 训练配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = RLHFTrainConfig()

        assert config.reward_model_epochs == 15  # 实际默认值
        assert config.ppo_epochs == 4
        assert config.clip_ratio == 0.2
        assert config.kl_coef == 0.01

    def test_custom_values(self):
        """测试自定义值"""
        config = RLHFTrainConfig(
            reward_model_epochs=5,
            ppo_lr=2e-5,
            clip_ratio=0.1
        )

        assert config.reward_model_epochs == 5
        assert config.ppo_lr == 2e-5
        assert config.clip_ratio == 0.1

    def test_ppo_parameters(self):
        """测试 PPO 参数"""
        config = RLHFTrainConfig()

        assert 0 < config.clip_ratio < 1
        assert config.kl_coef >= 0
        assert config.value_coef >= 0
        assert config.entropy_coef >= 0
        assert 0 < config.gamma <= 1
        assert 0 < config.lam <= 1


class TestRLVFTrainConfig:
    """RLVF 训练配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = RLVFTrainConfig()

        assert config.num_iterations == 60  # 实际默认值
        assert config.correct_reward == 1.0
        assert config.incorrect_reward == -0.5

    def test_custom_values(self):
        """测试自定义值"""
        config = RLVFTrainConfig(
            num_iterations=100,
            correct_reward=2.0,
            incorrect_reward=-1.0
        )

        assert config.num_iterations == 100
        assert config.correct_reward == 2.0
        assert config.incorrect_reward == -1.0

    def test_reward_difference(self):
        """测试奖励差异"""
        config = RLVFTrainConfig()
        # 正确奖励应该大于错误奖励
        assert config.correct_reward > config.incorrect_reward


class TestConfigConsistency:
    """配置一致性测试"""

    def test_config_reproducibility(self):
        """测试配置可重现性"""
        config1 = MyLLMConfig(
            vocab_size=1000,
            emb_dim=128
        )
        config2 = MyLLMConfig(
            vocab_size=1000,
            emb_dim=128
        )

        assert config1.vocab_size == config2.vocab_size
        assert config1.emb_dim == config2.emb_dim
        assert config1._estimate_params() == config2._estimate_params()

    def test_config_immutability_of_defaults(self):
        """测试默认值不被修改"""
        config1 = MyLLMConfig()
        vocab_size_before = config1.vocab_size

        config2 = MyLLMConfig(vocab_size=100)

        # config1 的值不应该被修改
        assert config1.vocab_size == vocab_size_before
        assert config2.vocab_size == 100


class TestConfigEdgeCases:
    """配置边界情况测试"""

    def test_min_config(self):
        """测试最小配置"""
        config = MyLLMConfig(
            vocab_size=10,
            emb_dim=8,
            num_heads=2,
            num_layers=1,
            context_size=4
        )
        assert config.head_dim == 4

    def test_large_vocab(self):
        """测试大词表"""
        config = MyLLMConfig(vocab_size=100000)
        assert config.vocab_size == 100000

    def test_zero_dropout(self):
        """测试零 dropout"""
        config = MyLLMConfig(dropout=0.0)
        assert config.dropout == 0.0

    def test_max_dropout(self):
        """测试最大 dropout"""
        config = MyLLMConfig(dropout=0.9)
        assert config.dropout == 0.9
