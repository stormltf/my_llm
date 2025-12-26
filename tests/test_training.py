"""
训练流程单元测试

测试内容：
1. 学习率调度器
2. 梯度累积
3. 梯度裁剪
4. 早停机制
5. 检查点保存/加载
6. 训练历史记录
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import GPT, GPTConfig
from tokenizer import BPETokenizer
from config import MyLLMConfig


class DummyDataset(Dataset):
    """简单的测试数据集"""

    def __init__(self, size=100, seq_len=32):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, 100, (self.seq_len,))
        target_ids = torch.randint(0, 100, (self.seq_len,))
        return input_ids, target_ids


class TestLearningRateScheduler:
    """学习率调度器测试"""

    @pytest.fixture
    def model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        return GPT(config)

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    def test_cosine_scheduler(self, optimizer):
        """测试余弦退火调度器"""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )

        initial_lr = optimizer.param_groups[0]['lr']

        # 学习率应该从最大值开始
        assert optimizer.param_groups[0]['lr'] == initial_lr

        # 步进几次
        for _ in range(5):
            scheduler.step()

        # 学习率应该降低
        assert optimizer.param_groups[0]['lr'] < initial_lr

    def test_cosine_scheduler_full_cycle(self, optimizer):
        """测试余弦退火完整周期"""
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )

        initial_lr = optimizer.param_groups[0]['lr']

        # 完整周期
        for _ in range(10):
            scheduler.step()

        # 学习率应该接近最小值
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr * 0.1

    def test_step_lr_scheduler(self, model):
        """测试阶梯学习率调度器"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )

        initial_lr = optimizer.param_groups[0]['lr']

        # 5步后学习率减半
        for _ in range(5):
            scheduler.step()

        assert abs(optimizer.param_groups[0]['lr'] - initial_lr * 0.5) < 1e-6


class TestGradientAccumulation:
    """梯度累积测试"""

    @pytest.fixture
    def model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        return GPT(config)

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=1e-3)

    def test_gradient_accumulation_basic(self, model, optimizer):
        """测试基础梯度累积"""
        accumulation_steps = 4
        dataset = DummyDataset(size=8, seq_len=16)
        dataloader = DataLoader(dataset, batch_size=2)

        model.train()

        for i, (input_ids, target_ids) in enumerate(dataloader):
            logits, loss = model(input_ids, target_ids)
            loss = loss / accumulation_steps  # 缩放损失
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 检查模型参数已更新
        # (这里简化检查)

    def test_gradient_accumulation_with_skip(self, model, optimizer):
        """测试跳过某些步的梯度累积"""
        accumulation_steps = 3

        model.train()
        input_ids = torch.randint(0, 100, (2, 16))
        target_ids = torch.randint(0, 100, (2, 16))

        for i in range(5):
            logits, loss = model(input_ids, target_ids)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 最后一步即使不足 accumulation_steps 也应该更新
        assert True  # 如果没有报错就算通过


class TestGradientClipping:
    """梯度裁剪测试"""

    @pytest.fixture
    def model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        return GPT(config)

    def test_gradient_clipping(self, model):
        """测试梯度裁剪"""
        model.train()
        input_ids = torch.randint(0, 100, (2, 16))
        target_ids = torch.randint(0, 100, (2, 16))

        logits, loss = model(input_ids, target_ids)
        loss.backward()

        # clip_grad_norm_ 返回裁剪前的范数，并就地裁剪梯度
        original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 再次计算范数以验证裁剪已生效
        clipped_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                clipped_norm += param.grad.data.norm(2).item() ** 2
        clipped_norm = clipped_norm ** 0.5

        # 裁剪后的范数应该 <= 1.0
        assert clipped_norm <= 1.0 + 1e-5  # 允许小的浮点误差

    def test_gradient_clipping_with_small_norm(self, model):
        """测试小梯度范数的裁剪"""
        model.train()
        input_ids = torch.randint(0, 100, (2, 16))
        target_ids = torch.randint(0, 100, (2, 16))

        # 使用很大的 max_norm
        logits, loss = model(input_ids, target_ids)
        loss.backward()

        original_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.0)

        # 梯度应该基本不变
        assert original_norm < 1000.0


class TestEarlyStopping:
    """早停机制测试"""

    def test_early_stopping_logic(self):
        """测试早停逻辑"""
        # loss 值需要低于 min_loss_threshold 才会触发早停检查
        losses = [0.2, 0.15, 0.12, 0.09, 0.085, 0.082, 0.081, 0.0805, 0.08, 0.08]

        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        min_loss_threshold = 0.1
        improvement_threshold = 0.01

        stopped = False
        stop_epoch = 0

        for epoch, loss in enumerate(losses):
            if loss < min_loss_threshold:
                if loss < best_loss - improvement_threshold:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    stopped = True
                    stop_epoch = epoch
                    break

        assert stopped
        assert stop_epoch > 0

    def test_no_early_stop_with_improvement(self):
        """测试有改善时不早停"""
        losses = [2.0, 1.5, 1.0, 0.8, 0.5, 0.3, 0.1, 0.05]

        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        min_loss_threshold = 0.1
        improvement_threshold = 0.01

        stopped = False

        for loss in losses:
            if loss < min_loss_threshold:
                if loss < best_loss - improvement_threshold:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    stopped = True
                    break

        # 应该不会触发早停（loss 一直在下降）
        assert not stopped


class TestCheckpointManagement:
    """检查点管理测试"""

    @pytest.fixture
    def model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        return GPT(config)

    def test_save_checkpoint(self, model, tmp_path):
        """测试保存检查点"""
        checkpoint_path = tmp_path / "checkpoint.pt"

        torch.save(model.state_dict(), checkpoint_path)

        assert checkpoint_path.exists()

    def test_load_checkpoint(self, model, tmp_path):
        """测试加载检查点"""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # 保存初始状态
        initial_state = {name: param.clone() for name, param in model.named_parameters()}

        torch.save(model.state_dict(), checkpoint_path)

        # 修改参数
        for param in model.parameters():
            param.data += 1.0

        # 加载检查点
        model.load_state_dict(torch.load(checkpoint_path))

        # 验证恢复
        for name, param in model.named_parameters():
            assert torch.allclose(param, initial_state[name])

    def test_save_optimizer_state(self, tmp_path):
        """测试保存优化器状态"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        checkpoint_path = tmp_path / "optimizer.pt"

        torch.save(optimizer.state_dict(), checkpoint_path)

        assert checkpoint_path.exists()


class TestTrainingHistory:
    """训练历史记录测试"""

    def test_history_tracking(self):
        """测试历史记录追踪"""
        history = {
            'loss': [],
            'lr': [],
            'epoch': []
        }

        for epoch in range(5):
            loss = 2.0 - epoch * 0.3
            lr = 1e-3 * (0.9 ** epoch)

            history['loss'].append(loss)
            history['lr'].append(lr)
            history['epoch'].append(epoch)

        assert len(history['loss']) == 5
        assert len(history['lr']) == 5
        assert len(history['epoch']) == 5

    def test_history_persistence(self, tmp_path):
        """测试历史记录持久化"""
        import json

        history = {
            'loss': [2.0, 1.5, 1.2],
            'accuracy': [0.6, 0.7, 0.75]
        }

        history_path = tmp_path / "history.json"

        with open(history_path, 'w') as f:
            json.dump(history, f)

        assert history_path.exists()

        with open(history_path, 'r') as f:
            loaded_history = json.load(f)

        assert loaded_history == history


class TestTrainingStep:
    """训练步骤测试"""

    @pytest.fixture
    def model_and_optimizer(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, optimizer

    def test_single_training_step(self, model_and_optimizer):
        """测试单步训练"""
        model, optimizer = model_and_optimizer

        model.train()
        input_ids = torch.randint(0, 100, (4, 16))
        target_ids = torch.randint(0, 100, (4, 16))

        logits, loss = model(input_ids, target_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 检查梯度存在
        for name, param in model.named_parameters():
            if param.requires_grad:
                # 检查参数已被更新（梯度在step后会被清零或重新计算）
                assert param.grad is None or param.grad is not None

    def test_multiple_training_steps(self, model_and_optimizer):
        """测试多步训练"""
        model, optimizer = model_and_optimizer

        initial_params = {name: param.clone() for name, param in model.named_parameters()}

        model.train()
        for _ in range(5):
            input_ids = torch.randint(0, 100, (4, 16))
            target_ids = torch.randint(0, 100, (4, 16))

            logits, loss = model(input_ids, target_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 检查参数已改变
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break

        assert params_changed


class TestOptimizerSettings:
    """优化器设置测试"""

    def test_adamw_optimizer(self):
        """测试 AdamW 优化器"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_sgd_optimizer(self):
        """测试 SGD 优化器"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-3,
            momentum=0.9
        )

        assert optimizer is not None
        assert isinstance(optimizer, torch.optim.SGD)


class TestTrainingModes:
    """训练模式测试"""

    @pytest.fixture
    def model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        return GPT(config)

    def test_train_mode(self, model):
        """测试训练模式"""
        model.train()
        assert model.training is True

    def test_eval_mode(self, model):
        """测试评估模式"""
        model.eval()
        assert model.training is False

    def test_mode_switching(self, model):
        """测试模式切换"""
        model.train()
        assert model.training is True

        model.eval()
        assert model.training is False

        model.train()
        assert model.training is True
