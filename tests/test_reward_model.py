"""
奖励模型单元测试

测试内容：
1. RewardModel 架构
2. 前向传播
3. 损失计算
4. 从预训练模型初始化
5. RewardDataset
"""

import pytest
import torch
import torch.nn.functional as F

from config import MyLLMConfig, get_mini_config
from model import GPT, GPTConfig
from reward_model import RewardModel, RewardDataset, RewardModelTrainer


class TestRewardModel:
    """奖励模型测试"""

    @pytest.fixture
    def reward_config(self):
        """小型测试配置"""
        return MyLLMConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )

    @pytest.fixture
    def reward_model(self, reward_config):
        """创建奖励模型"""
        return RewardModel(reward_config)

    def test_model_creation(self, reward_config):
        """测试模型创建"""
        model = RewardModel(reward_config)
        assert model is not None

    def test_model_components(self, reward_model):
        """测试模型组件"""
        assert hasattr(reward_model, 'token_embedding')
        assert hasattr(reward_model, 'position_embedding')
        assert hasattr(reward_model, 'transformer_blocks')
        assert hasattr(reward_model, 'final_norm')
        assert hasattr(reward_model, 'reward_head')

    def test_forward_output_shape(self, reward_model):
        """测试前向传播输出形状"""
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        rewards = reward_model(input_ids)

        # 输出应该是每个样本一个标量
        assert rewards.shape == (batch_size,)

    def test_forward_with_attention_mask(self, reward_model):
        """测试带注意力掩码的前向传播"""
        batch_size = 4
        seq_len = 16
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 8:] = 0  # 第一个样本只有 8 个有效 token

        rewards = reward_model(input_ids, attention_mask)

        assert rewards.shape == (batch_size,)
        assert not torch.isnan(rewards).any()

    def test_forward_no_nan(self, reward_model):
        """测试前向传播无 NaN"""
        input_ids = torch.randint(0, 100, (2, 10))
        rewards = reward_model(input_ids)

        assert not torch.isnan(rewards).any()
        assert not torch.isinf(rewards).any()

    def test_num_parameters(self, reward_model):
        """测试参数量"""
        num_params = reward_model.num_parameters()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_reward_head_architecture(self, reward_model):
        """测试奖励头架构"""
        # 奖励头应该是一个 Sequential
        assert isinstance(reward_model.reward_head, torch.nn.Sequential)

        # 最后一层输出维度应该是 1
        last_layer = reward_model.reward_head[-1]
        assert last_layer.out_features == 1


class TestRewardModelFromPretrained:
    """从预训练模型初始化测试"""

    @pytest.fixture
    def base_model_and_config(self):
        """创建基础模型和配置"""
        config = MyLLMConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )

        # 创建 GPT 模型
        gpt_config = GPTConfig(
            vocab_size=config.vocab_size,
            emb_dim=config.emb_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            context_size=config.context_size,
            dropout=config.dropout
        )
        base_model = GPT(gpt_config)

        return base_model, config

    def test_from_pretrained(self, base_model_and_config):
        """测试从预训练模型初始化"""
        base_model, config = base_model_and_config
        reward_model = RewardModel.from_pretrained(base_model, config)

        assert reward_model is not None

    def test_weights_copied(self, base_model_and_config):
        """测试权重是否正确复制"""
        base_model, config = base_model_and_config
        reward_model = RewardModel.from_pretrained(base_model, config)

        # Token embedding 权重应该相同
        assert torch.allclose(
            reward_model.token_embedding.weight,
            base_model.token_embedding.weight
        )

        # Position embedding 权重应该相同
        assert torch.allclose(
            reward_model.position_embedding.weight,
            base_model.position_embedding.weight
        )


class TestBradleyTerryLoss:
    """Bradley-Terry 损失测试"""

    def test_loss_computation(self):
        """测试损失计算"""
        chosen_rewards = torch.tensor([1.0, 2.0, 1.5])
        rejected_rewards = torch.tensor([0.5, 1.0, 0.5])

        # L = -log(sigmoid(r_chosen - r_rejected))
        reward_diff = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(reward_diff).mean()

        # 当 chosen > rejected 时，损失应该较小
        assert loss.item() < 1.0

    def test_loss_increases_when_rejected_higher(self):
        """测试当 rejected 分数更高时损失增加"""
        # 正确排序
        chosen_high = torch.tensor([2.0])
        rejected_low = torch.tensor([1.0])
        loss_correct = -F.logsigmoid(chosen_high - rejected_low).mean()

        # 错误排序
        chosen_low = torch.tensor([1.0])
        rejected_high = torch.tensor([2.0])
        loss_wrong = -F.logsigmoid(chosen_low - rejected_high).mean()

        assert loss_wrong > loss_correct

    def test_loss_gradient(self):
        """测试损失梯度"""
        chosen_rewards = torch.tensor([1.0], requires_grad=True)
        rejected_rewards = torch.tensor([0.5], requires_grad=True)

        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        loss.backward()

        # chosen 的梯度应该是负的（增加 chosen 会减小损失）
        assert chosen_rewards.grad.item() < 0
        # rejected 的梯度应该是正的
        assert rejected_rewards.grad.item() > 0


class TestRewardDataset:
    """奖励数据集测试"""

    @pytest.fixture
    def mock_tokenizer(self):
        """模拟分词器"""
        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0

            def encode(self, text):
                # 简单的模拟编码
                return [ord(c) % 100 for c in text[:50]]

        return MockTokenizer()

    @pytest.fixture
    def sample_data(self):
        """示例数据"""
        return [
            {
                "prompt": "什么是人工智能？",
                "chosen": "人工智能是计算机科学的一个重要分支。",
                "rejected": "不知道。"
            },
            {
                "prompt": "你好",
                "chosen": "你好！有什么可以帮助你的？",
                "rejected": "嗯"
            }
        ]

    def test_dataset_creation(self, sample_data, mock_tokenizer):
        """测试数据集创建"""
        dataset = RewardDataset(sample_data, mock_tokenizer, max_length=64)
        assert len(dataset) == 2

    def test_dataset_item(self, sample_data, mock_tokenizer):
        """测试数据集项"""
        dataset = RewardDataset(sample_data, mock_tokenizer, max_length=64)
        item = dataset[0]

        assert 'chosen_ids' in item
        assert 'rejected_ids' in item
        assert 'chosen_mask' in item
        assert 'rejected_mask' in item

    def test_dataset_item_shape(self, sample_data, mock_tokenizer):
        """测试数据集项形状"""
        max_length = 64
        dataset = RewardDataset(sample_data, mock_tokenizer, max_length=max_length)
        item = dataset[0]

        assert item['chosen_ids'].shape == (max_length,)
        assert item['rejected_ids'].shape == (max_length,)
        assert item['chosen_mask'].shape == (max_length,)
        assert item['rejected_mask'].shape == (max_length,)

    def test_attention_mask_values(self, sample_data, mock_tokenizer):
        """测试注意力掩码值"""
        dataset = RewardDataset(sample_data, mock_tokenizer, max_length=64)
        item = dataset[0]

        # mask 值应该只有 0 和 1
        assert torch.all((item['chosen_mask'] == 0) | (item['chosen_mask'] == 1))
        assert torch.all((item['rejected_mask'] == 0) | (item['rejected_mask'] == 1))


class TestRewardModelTraining:
    """奖励模型训练测试"""

    @pytest.fixture
    def training_setup(self):
        """训练设置"""
        config = MyLLMConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=1,
            context_size=32,
            dropout=0.0
        )

        model = RewardModel(config)

        class MockTokenizer:
            def __init__(self):
                self.pad_token_id = 0

            def encode(self, text):
                return [ord(c) % 100 for c in text[:30]]

        tokenizer = MockTokenizer()

        return model, tokenizer, config

    def test_training_step(self, training_setup):
        """测试训练步骤"""
        model, tokenizer, config = training_setup

        # 创建一个批次
        batch_size = 2
        seq_len = 16
        chosen_ids = torch.randint(0, 100, (batch_size, seq_len))
        rejected_ids = torch.randint(0, 100, (batch_size, seq_len))

        # 前向传播
        chosen_rewards = model(chosen_ids)
        rejected_rewards = model(rejected_ids)

        # 计算损失
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

        # 反向传播
        loss.backward()

        # 检查梯度存在
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_accuracy_metric(self, training_setup):
        """测试准确率指标"""
        model, _, _ = training_setup

        chosen_ids = torch.randint(0, 100, (4, 16))
        rejected_ids = torch.randint(0, 100, (4, 16))

        chosen_rewards = model(chosen_ids)
        rejected_rewards = model(rejected_ids)

        # 计算准确率
        correct = (chosen_rewards > rejected_rewards).sum().item()
        accuracy = correct / len(chosen_rewards)

        # 准确率应该在 0-1 之间
        assert 0 <= accuracy <= 1


class TestRewardModelEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def reward_model(self):
        config = MyLLMConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=1,
            context_size=32,
            dropout=0.0
        )
        return RewardModel(config)

    def test_single_token_input(self, reward_model):
        """测试单 token 输入"""
        input_ids = torch.tensor([[5]])
        rewards = reward_model(input_ids)
        assert rewards.shape == (1,)

    def test_batch_size_one(self, reward_model):
        """测试批次大小为 1"""
        input_ids = torch.randint(0, 100, (1, 10))
        rewards = reward_model(input_ids)
        assert rewards.shape == (1,)

    def test_all_same_tokens(self, reward_model):
        """测试所有 token 相同"""
        input_ids = torch.ones(2, 10, dtype=torch.long)
        rewards = reward_model(input_ids)

        # 相同输入应该产生相同输出
        assert torch.allclose(rewards[0], rewards[1])

    def test_different_sequence_lengths_with_mask(self, reward_model):
        """测试不同序列长度（使用掩码）"""
        batch_size = 2
        max_len = 20

        input_ids = torch.randint(0, 100, (batch_size, max_len))
        attention_mask = torch.ones(batch_size, max_len)
        attention_mask[0, 10:] = 0  # 第一个序列长度 10
        attention_mask[1, 15:] = 0  # 第二个序列长度 15

        rewards = reward_model(input_ids, attention_mask)

        assert rewards.shape == (batch_size,)
        assert not torch.isnan(rewards).any()
