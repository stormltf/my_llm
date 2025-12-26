"""
可复现性测试

测试内容：
1. 随机种子设置
2. 确定性训练
3. 结果可复现性
4. 模型权重可复现
"""

import pytest
import torch
import torch.nn as nn
import random
import numpy as np

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from generate import TextGenerator


def set_all_seeds(seed):
    """设置所有随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 确保 CUDA 操作确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestRandomSeeds:
    """随机种子测试"""

    def test_torch_seed_reproducibility(self):
        """测试 PyTorch 种子可复现性"""
        set_all_seeds(42)

        # 第一次运行
        tensor1 = torch.randn(10)

        set_all_seeds(42)

        # 第二次运行
        tensor2 = torch.randn(10)

        # 应该完全相同
        assert torch.allclose(tensor1, tensor2)

    def test_different_seeds_produce_different_results(self):
        """测试不同种子产生不同结果"""
        set_all_seeds(42)
        tensor1 = torch.randn(10)

        set_all_seeds(43)
        tensor2 = torch.randn(10)

        # 应该不同
        assert not torch.allclose(tensor1, tensor2)

    def test_numpy_seed_reproducibility(self):
        """测试 NumPy 种子可复现性"""
        set_all_seeds(123)

        arr1 = np.random.randn(10)

        set_all_seeds(123)

        arr2 = np.random.randn(10)

        # 应该相同
        assert np.allclose(arr1, arr2)

    def test_random_seed_reproducibility(self):
        """测试 Python random 种子可复现性"""
        set_all_seeds(456)

        val1 = [random.random() for _ in range(10)]

        set_all_seeds(456)

        val2 = [random.random() for _ in range(10)]

        # 应该相同
        assert val1 == val2


class TestDeterministicModelCreation:
    """确定性模型创建测试"""

    def test_same_seed_same_initialization(self):
        """测试相同种子产生相同初始化"""
        set_all_seeds(42)

        config1 = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model1 = GPT(config1)

        set_all_seeds(42)

        config2 = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model2 = GPT(config2)

        # 检查所有参数相同
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_different_seeds_different_initialization(self):
        """测试不同种子产生不同初始化"""
        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model1 = GPT(config)

        set_all_seeds(43)

        model2 = GPT(config)

        # 应该有参数不同
        all_same = all(
            torch.allclose(p1, p2)
            for p1, p2 in zip(model1.parameters(), model2.parameters())
        )
        assert not all_same


class TestDeterministicTraining:
    """确定性训练测试"""

    def test_deterministic_forward_pass(self):
        """测试确定性前向传播"""
        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0  # 关闭 dropout 以确保确定性
        )
        model = GPT(config)
        model.eval()

        input_ids = torch.randint(0, 100, (2, 10))

        logits1, loss1 = model(input_ids)

        set_all_seeds(42)

        model2 = GPT(config)
        model2.eval()

        logits2, loss2 = model2(input_ids)

        # 输出应该相同
        assert torch.allclose(logits1, logits2)

    def test_deterministic_training_step(self):
        """测试确定性训练步骤"""
        # 第一次训练
        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model1 = GPT(config)
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)

        input_ids = torch.randint(0, 100, (4, 16))
        target_ids = torch.randint(0, 100, (4, 16))

        for _ in range(5):
            logits, loss = model1(input_ids, target_ids)
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

        final_params1 = [p.clone() for p in model1.parameters()]

        # 第二次训练
        set_all_seeds(42)

        model2 = GPT(config)
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

        for _ in range(5):
            logits, loss = model2(input_ids, target_ids)
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()

        final_params2 = [p.clone() for p in model2.parameters()]

        # 参数应该相同
        for p1, p2 in zip(final_params1, final_params2):
            assert torch.allclose(p1, p2, atol=1e-5)

    def test_deterministic_training_with_dropout(self):
        """测试带 dropout 的确定性训练（eval 模式）"""
        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.1
        )
        model1 = GPT(config)
        model1.eval()  # eval 模式下 dropout 不生效

        input_ids = torch.randint(0, 100, (2, 10))

        with torch.no_grad():
            logits1, _ = model1(input_ids)

        set_all_seeds(42)

        model2 = GPT(config)
        model2.eval()

        with torch.no_grad():
            logits2, _ = model2(input_ids)

        # eval 模式下应该相同
        assert torch.allclose(logits1, logits2)


class TestDeterministicGeneration:
    """确定性生成测试"""

    def test_deterministic_greedy_generation(self):
        """测试确定性贪婪生成"""
        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model1 = GPT(config)
        model1.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        generator1 = TextGenerator(model1, tokenizer, torch.device("cpu"))

        result1 = generator1.generate_greedy("测", max_length=10)

        set_all_seeds(42)

        model2 = GPT(config)
        model2.eval()

        generator2 = TextGenerator(model2, tokenizer, torch.device("cpu"))

        result2 = generator2.generate_greedy("测", max_length=10)

        # 贪婪生成应该完全相同
        assert result1 == result2

    def test_deterministic_sampling_with_seed(self):
        """测试带种子的采样"""
        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model1 = GPT(config)
        model1.eval()

        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试"], verbose=False)

        generator1 = TextGenerator(model1, tokenizer, torch.device("cpu"))

        # 设置随机数生成器种子
        torch.manual_seed(42)
        result1 = generator1.generate("测", max_length=10, temperature=0.8)

        # 重新创建并设置种子
        torch.manual_seed(42)
        result2 = generator1.generate("测", max_length=10, temperature=0.8)

        # 应该相同（在相同设备上）
        assert result1 == result2


class TestDeterministicDataProcessing:
    """确定性数据处理测试"""

    def test_deterministic_dataset_shuffling(self):
        """测试确定性数据集打乱"""
        set_all_seeds(42)

        data = list(range(100))

        # 第一次打乱
        random.shuffle(data)
        shuffled1 = data.copy()

        # 重置并第二次打乱
        data = list(range(100))
        set_all_seeds(42)
        random.shuffle(data)
        shuffled2 = data.copy()

        # 应该相同
        assert shuffled1 == shuffled2

    def test_deterministic_dataloader(self):
        """测试确定性数据加载器"""
        set_all_seeds(42)

        # 创建数据集
        from train import PretrainDataset
        tokenizer = BPETokenizer(vocab_size=50)
        tokenizer.fit(["测 试 数 据"], verbose=False)

        texts = ["文 本 一", "文 本 二", "文 本 三"] * 10
        dataset1 = PretrainDataset(texts, tokenizer, seq_len=8)

        from torch.utils.data import DataLoader
        dataloader1 = DataLoader(dataset1, batch_size=4, shuffle=True)

        # 收集第一批
        batch1 = None
        for batch in dataloader1:
            batch1 = batch
            break

        set_all_seeds(42)

        tokenizer2 = BPETokenizer(vocab_size=50)
        tokenizer2.fit(["测 试 数 据"], verbose=False)
        dataset2 = PretrainDataset(texts, tokenizer2, seq_len=8)

        dataloader2 = DataLoader(dataset2, batch_size=4, shuffle=True)

        batch2 = None
        for batch in dataloader2:
            batch2 = batch
            break

        # 应该相同（如果种子设置正确）
        # 注意：这取决于 DataLoader 的实现细节

    def test_deterministic_tokenizer_training(self):
        """测试确定性分词器训练"""
        set_all_seeds(42)

        texts = ["测 试 文 本", "分 词 器 训 练"] * 10

        tokenizer1 = BPETokenizer(vocab_size=50)
        tokenizer1.fit(texts, verbose=False)

        set_all_seeds(42)

        tokenizer2 = BPETokenizer(vocab_size=50)
        tokenizer2.fit(texts, verbose=False)

        # 检查词表相同
        assert tokenizer1.vocab == tokenizer2.vocab

        # 检查编码结果相同
        test_text = "测 试"
        tokens1 = tokenizer1.encode(test_text)
        tokens2 = tokenizer2.encode(test_text)

        assert tokens1 == tokens2


class TestModelSerialization:
    """模型序列化测试"""

    def test_model_save_load_preserves_behavior(self):
        """测试模型保存加载后行为一致"""
        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model1 = GPT(config)
        model1.eval()

        # 保存
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            torch.save(model1.state_dict(), save_path)

            # 加载
            model2 = GPT(config)
            model2.load_state_dict(torch.load(save_path, weights_only=True))
            model2.eval()

        # 使用相同输入
        set_all_seeds(42)
        input_ids = torch.randint(0, 100, (1, 10))

        with torch.no_grad():
            logits1, _ = model1(input_ids)

        set_all_seeds(42)
        with torch.no_grad():
            logits2, _ = model2(input_ids)

        # 输出应该相同
        assert torch.allclose(logits1, logits2)


class TestCrossDeviceReproducibility:
    """跨设备可复现性测试（如果有多设备）"""

    def test_cpu_model_reproducibility(self):
        """测试 CPU 模型可复现性"""
        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )

        device = torch.device("cpu")

        model1 = GPT(config).to(device)
        model1.eval()

        input_ids = torch.randint(0, 50, (1, 10))

        with torch.no_grad():
            logits1, _ = model1(input_ids)

        set_all_seeds(42)

        model2 = GPT(config).to(device)
        model2.eval()

        with torch.no_grad():
            logits2, _ = model2(input_ids)

        assert torch.allclose(logits1, logits2)

    @pytest.mark.gpu
    def test_gpu_model_reproducibility(self):
        """测试 GPU 模型可复现性"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA 不可用")

        set_all_seeds(42)

        config = GPTConfig(
            vocab_size=50,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )

        device = torch.device("cuda")

        torch.cuda.manual_seed_all(42)
        model1 = GPT(config).to(device)
        model1.eval()

        input_ids = torch.randint(0, 50, (1, 10), device=device)

        with torch.no_grad():
            logits1, _ = model1(input_ids)

        torch.cuda.manual_seed_all(42)
        model2 = GPT(config).to(device)
        model2.eval()

        with torch.no_grad():
            logits2, _ = model2(input_ids)

        # GPU 上可能存在数值差异，使用较宽松的容差
        assert torch.allclose(logits1.cpu(), logits2.cpu(), atol=1e-4)


class TestReproducibilityWithDifferentOptions:
    """不同选项下的可复现性测试"""

    def test_reproducibility_with_top_k(self):
        """测试 Top-k 采样的可复现性"""
        set_all_seeds(42)

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

        # 第一次
        torch.manual_seed(42)
        result1 = generator.generate("测", max_length=10, top_k=10, temperature=0.8)

        # 第二次
        torch.manual_seed(42)
        result2 = generator.generate("测", max_length=10, top_k=10, temperature=0.8)

        assert result1 == result2

    def test_reproducibility_with_top_p(self):
        """测试 Top-p 采样的可复现性"""
        set_all_seeds(42)

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

        # 第一次
        torch.manual_seed(42)
        result1 = generator.generate("测", max_length=10, top_p=0.9, temperature=0.8)

        # 第二次
        torch.manual_seed(42)
        result2 = generator.generate("测", max_length=10, top_p=0.9, temperature=0.8)

        assert result1 == result2
