"""
数据集单元测试

测试内容：
1. PretrainDataset - 预训练数据集
2. SFTDataset - 监督微调数据集
3. collate_fn - 批次整理函数
4. 数据加载功能
"""

import pytest
import torch
from torch.utils.data import DataLoader

from train import (
    PretrainDataset,
    SFTDataset,
    collate_fn,
    load_pretrain_data,
    load_sft_data,
    load_reward_data,
    load_rlvf_data
)
from tokenizer import BPETokenizer


class TestPretrainDataset:
    """预训练数据集测试"""

    @pytest.fixture
    def sample_texts(self):
        """示例训练文本"""
        return [
            "我 喜欢 学习 人工智能",
            "人工智能 是 未来 的 趋势",
            "深度 学习 是 机器 学习 的 分支",
        ]

    @pytest.fixture
    def tokenizer(self):
        """创建分词器"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["我 喜欢 学习", "人工智能 很 有趣", "深度 学习 强大"]
        tokenizer.fit(texts, verbose=False)
        return tokenizer

    @pytest.fixture
    def dataset(self, sample_texts, tokenizer):
        """创建预训练数据集"""
        return PretrainDataset(sample_texts, tokenizer, seq_len=8)

    def test_dataset_creation(self, dataset):
        """测试数据集创建"""
        assert dataset is not None
        assert len(dataset) > 0

    def test_dataset_length(self, dataset):
        """测试数据集长度"""
        # 序列长度为 8，样本数应该大于 0
        assert len(dataset) > 0

    def test_getitem_shape(self, dataset):
        """测试获取样本的形状"""
        input_ids, target_ids = dataset[0]

        # 检查形状
        assert input_ids.shape == (8,)
        assert target_ids.shape == (8,)

        # 检查类型
        assert input_ids.dtype == torch.long
        assert target_ids.dtype == torch.long

    def test_autoregressive_property(self, dataset):
        """测试自回归属性"""
        input_ids, target_ids = dataset[0]

        # 目标应该是输入的下一个词（偏移1位）
        # 即 target[i] 应该等于 input[i+1]
        assert torch.equal(target_ids[:-1], input_ids[1:])

    def test_dataloader(self, dataset):
        """测试数据加载器"""
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        for input_ids, target_ids in dataloader:
            # 检查批次形状
            assert input_ids.shape[0] <= 2
            assert input_ids.shape[1] == 8
            assert target_ids.shape == input_ids.shape
            break  # 只测试第一个批次

    def test_empty_texts(self, tokenizer):
        """测试空文本列表"""
        dataset = PretrainDataset([], tokenizer, seq_len=8)
        assert len(dataset) == 0

    def test_single_token_texts(self, tokenizer):
        """测试单 token 文本"""
        dataset = PretrainDataset(["我"], tokenizer, seq_len=8)
        # 序列长度不足，无法生成样本
        assert len(dataset) == 0


class TestSFTDataset:
    """SFT 数据集测试"""

    @pytest.fixture
    def sample_data(self):
        """示例 SFT 数据"""
        return [
            {"user": "你好", "assistant": "你好！有什么可以帮助你的吗？"},
            {"user": "1+1等于多少", "assistant": "1+1等于2"},
            {"user": "什么是人工智能", "assistant": "人工智能是计算机科学的一个分支"},
        ]

    @pytest.fixture
    def tokenizer(self):
        """创建分词器"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["你 好", "1 + 1 = 2", "人工 智能 是 科学"]
        tokenizer.fit(texts, verbose=False)
        return tokenizer

    @pytest.fixture
    def dataset(self, sample_data, tokenizer):
        """创建 SFT 数据集"""
        return SFTDataset(sample_data, tokenizer, max_length=64)

    def test_dataset_creation(self, dataset):
        """测试数据集创建"""
        assert dataset is not None
        assert len(dataset) > 0

    def test_dataset_length(self, dataset):
        """测试数据集长度"""
        # 应该有与输入数据相同数量的样本
        assert len(dataset) > 0

    def test_getitem_shape(self, dataset):
        """测试获取样本的形状"""
        input_ids, target_ids = dataset[0]

        # 输入和目标应该长度相同
        assert input_ids.shape == target_ids.shape
        assert input_ids.dim() == 1

    def test_loss_mask_present(self, dataset):
        """测试 loss mask 存在"""
        _, target_ids = dataset[0]

        # PyTorch 默认使用 -100 作为 ignore_index
        target_list = target_ids.tolist()
        assert -100 in target_list, f"Expected -100 in target_ids, got {target_list}"

    def test_dataloader_with_collate(self, dataset):
        """测试带 collate_fn 的数据加载器"""
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn
        )

        for input_ids, target_ids in dataloader:
            # 检查批次形状
            assert input_ids.shape[0] <= 2
            assert input_ids.shape == target_ids.shape
            break

    def test_empty_data(self, tokenizer):
        """测试空数据"""
        dataset = SFTDataset([], tokenizer, max_length=64)
        assert len(dataset) == 0


class TestCollateFn:
    """批次整理函数测试"""

    def test_collate_fn_basic(self):
        """测试基础 collate_fn"""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
            (torch.tensor([5, 6]), torch.tensor([6, 7])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        # 检查形状
        assert padded_inputs.shape == (2, 3)
        assert padded_targets.shape == (2, 3)

    def test_collate_fn_padding(self):
        """测试填充"""
        batch = [
            (torch.tensor([1, 2]), torch.tensor([2, 3])),
            (torch.tensor([4, 5, 6, 7]), torch.tensor([5, 6, 7, 8])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        # 第一个样本应该被填充
        assert padded_inputs[0, 2].item() == 0  # input 填充 0
        # PyTorch 默认使用 -100 作为 ignore_index
        assert padded_targets[0, 2].item() == -100  # target 填充 ignore_index

    def test_collate_fn_single_batch(self):
        """测试单样本批次"""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        ]

        padded_inputs, padded_targets = collate_fn(batch)

        assert padded_inputs.shape == (1, 3)
        assert padded_targets.shape == (1, 3)

    def test_collate_fn_empty_batch(self):
        """测试空批次"""
        with pytest.raises((ValueError, RuntimeError)):
            collate_fn([])


class TestDataLoaders:
    """数据加载功能测试"""

    def test_load_pretrain_data(self):
        """测试加载预训练数据"""
        data = load_pretrain_data()

        assert isinstance(data, list)
        if data:
            assert isinstance(data[0], str)

    def test_load_sft_data(self):
        """测试加载 SFT 数据"""
        data = load_sft_data()

        assert isinstance(data, list)
        if data:
            assert 'user' in data[0]
            assert 'assistant' in data[0]

    def test_load_reward_data(self):
        """测试加载奖励数据"""
        data = load_reward_data()

        assert isinstance(data, list)

    def test_load_rlvf_data(self):
        """测试加载 RLVF 数据"""
        data = load_rlvf_data()

        assert isinstance(data, list)


class TestDatasetIntegration:
    """数据集集成测试"""

    @pytest.fixture
    def tokenizer(self):
        """创建分词器"""
        tokenizer = BPETokenizer(vocab_size=100)
        texts = ["测试 文本 数据", "集成 测试 场景"]
        tokenizer.fit(texts, verbose=False)
        return tokenizer

    def test_full_training_loop(self, tokenizer):
        """测试完整训练循环"""
        # 创建数据集
        texts = ["测试 数据 一", "测试 数据 二", "测试 数据 三"] * 10
        dataset = PretrainDataset(texts, tokenizer, seq_len=16)

        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        batch_count = 0
        for input_ids, target_ids in dataloader:
            assert input_ids.shape[1] == 16
            assert target_ids.shape[1] == 16
            batch_count += 1
            if batch_count >= 3:
                break

        assert batch_count > 0

    def test_sft_training_loop(self, tokenizer):
        """测试 SFT 训练循环"""
        data = [
            {"user": "你好", "assistant": "你好！"},
            {"user": "再见", "assistant": "再见！"},
        ] * 5

        dataset = SFTDataset(data, tokenizer, max_length=32)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        batch_count = 0
        for input_ids, target_ids in dataloader:
            assert input_ids.shape[0] <= 2
            batch_count += 1
            if batch_count >= 3:
                break

        assert batch_count > 0


class TestDatasetEdgeCases:
    """数据集边界情况测试"""

    @pytest.fixture
    def tokenizer(self):
        """创建分词器"""
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试", "边 界"], verbose=False)
        return tokenizer

    def test_very_long_sequence(self, tokenizer):
        """测试超长序列"""
        data = [{"user": "问" * 100, "assistant": "答" * 100}]
        dataset = SFTDataset(data, tokenizer, max_length=32)

        # 应该被截断
        input_ids, target_ids = dataset[0]
        assert len(input_ids) <= 32

    def test_unicode_characters(self, tokenizer):
        """测试 Unicode 字符"""
        data = [
            {"user": "Hello 世界", "assistant": "你好"},
            {"user": "🚀 rocket", "assistant": "火箭"},
        ]

        dataset = SFTDataset(data, tokenizer, max_length=32)
        assert len(dataset) > 0

    def test_special_tokens(self, tokenizer):
        """测试特殊 token"""
        data = [
            {"user": "<|im_start|>test", "assistant": "<|im_end|>reply"},
        ]

        dataset = SFTDataset(data, tokenizer, max_length=32)
        assert len(dataset) > 0
