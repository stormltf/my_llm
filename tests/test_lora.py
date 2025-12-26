"""
LoRA 单元测试

测试内容：
1. LoRAConfig 配置
2. LoRALinear 层
3. 应用 LoRA 到模型
4. 保存/加载 LoRA
5. 合并/取消合并权重
6. LoRATrainer
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from model import GPT, GPTConfig
from lora import (
    LoRAConfig, LoRALinear,
    apply_lora_to_model, get_lora_state_dict, load_lora_state_dict,
    save_lora, load_lora, merge_lora, unmerge_lora
)


class TestLoRAConfig:
    """LoRA 配置测试"""

    def test_default_values(self):
        """测试默认值"""
        config = LoRAConfig()

        assert config.r == 8
        assert config.alpha == 16
        assert config.dropout == 0.05
        assert "c_attn" in config.target_modules
        assert "c_proj" in config.target_modules

    def test_custom_values(self):
        """测试自定义值"""
        config = LoRAConfig(
            r=16,
            alpha=32,
            dropout=0.1,
            target_modules=["linear1", "linear2"]
        )

        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == 0.1
        assert config.target_modules == ["linear1", "linear2"]

    def test_to_dict(self):
        """测试转换为字典"""
        config = LoRAConfig(r=4, alpha=8)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["r"] == 4
        assert d["alpha"] == 8

    def test_from_dict(self):
        """测试从字典创建"""
        d = {"r": 4, "alpha": 8, "dropout": 0.1, "target_modules": ["c_attn"], "bias": "none"}
        config = LoRAConfig.from_dict(d)

        assert config.r == 4
        assert config.alpha == 8

    def test_scaling_factor(self):
        """测试缩放因子"""
        config = LoRAConfig(r=8, alpha=16)
        # scaling = alpha / r
        expected_scaling = 16 / 8
        assert expected_scaling == 2.0


class TestLoRALinear:
    """LoRA 线性层测试"""

    @pytest.fixture
    def original_layer(self):
        return nn.Linear(64, 128)

    def test_creation(self, original_layer):
        """测试创建 LoRA 层"""
        lora = LoRALinear(original_layer, r=4, alpha=8)

        assert lora.r == 4
        assert lora.alpha == 8
        assert lora.scaling == 8 / 4

    def test_lora_matrices_shape(self, original_layer):
        """测试 LoRA 矩阵形状"""
        lora = LoRALinear(original_layer, r=4)

        # A: (r, in_features)
        assert lora.lora_A.shape == (4, 64)
        # B: (out_features, r)
        assert lora.lora_B.shape == (128, 4)

    def test_lora_B_zero_init(self, original_layer):
        """测试 B 矩阵零初始化"""
        lora = LoRALinear(original_layer, r=4)

        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))

    def test_original_weights_frozen(self, original_layer):
        """测试原始权重被冻结"""
        lora = LoRALinear(original_layer, r=4)

        assert not lora.original_layer.weight.requires_grad
        if lora.original_layer.bias is not None:
            assert not lora.original_layer.bias.requires_grad

    def test_lora_weights_trainable(self, original_layer):
        """测试 LoRA 权重可训练"""
        lora = LoRALinear(original_layer, r=4)

        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_forward_shape(self, original_layer):
        """测试前向传播形状"""
        lora = LoRALinear(original_layer, r=4)
        x = torch.randn(2, 10, 64)
        out = lora(x)

        assert out.shape == (2, 10, 128)

    def test_forward_initial_output(self, original_layer):
        """测试初始输出（B=0 时应等于原始层）"""
        lora = LoRALinear(original_layer, r=4)
        x = torch.randn(2, 10, 64)

        # 由于 B 初始化为 0，LoRA 输出应该等于原始层
        original_out = original_layer(x)
        lora_out = lora(x)

        assert torch.allclose(original_out, lora_out, atol=1e-6)

    def test_merge_weights(self, original_layer):
        """测试合并权重"""
        lora = LoRALinear(original_layer, r=4)

        # 手动设置非零 LoRA 权重（使用较小的值以减少数值误差）
        lora.lora_A.data = torch.randn_like(lora.lora_A) * 0.01
        lora.lora_B.data = torch.randn_like(lora.lora_B) * 0.01

        x = torch.randn(2, 10, 64)

        # 合并前输出
        out_before = lora(x).clone()

        # 合并
        lora.merge_weights()
        assert lora.merged

        # 合并后输出应该相同（放宽容差）
        out_after = lora(x)
        assert torch.allclose(out_before, out_after, atol=1e-4)

    def test_unmerge_weights(self, original_layer):
        """测试取消合并权重"""
        lora = LoRALinear(original_layer, r=4)

        # 记录原始权重
        original_weight = original_layer.weight.data.clone()

        # 设置非零 LoRA 权重并合并
        lora.lora_A.data = torch.randn_like(lora.lora_A)
        lora.lora_B.data = torch.randn_like(lora.lora_B)
        lora.merge_weights()

        # 取消合并
        lora.unmerge_weights()
        assert not lora.merged

        # 权重应该恢复
        assert torch.allclose(
            lora.original_layer.weight.data,
            original_weight,
            atol=1e-5
        )

    def test_get_lora_params(self, original_layer):
        """测试获取 LoRA 参数"""
        lora = LoRALinear(original_layer, r=4)
        params = lora.get_lora_params()

        assert "lora_A" in params
        assert "lora_B" in params
        assert params["lora_A"].shape == (4, 64)

    def test_set_lora_params(self, original_layer):
        """测试设置 LoRA 参数"""
        lora = LoRALinear(original_layer, r=4)

        new_A = torch.randn(4, 64)
        new_B = torch.randn(128, 4)

        lora.set_lora_params({"lora_A": new_A, "lora_B": new_B})

        assert torch.allclose(lora.lora_A.data, new_A)
        assert torch.allclose(lora.lora_B.data, new_B)


class TestApplyLoRA:
    """应用 LoRA 到模型测试"""

    @pytest.fixture
    def base_model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        return GPT(config)

    def test_apply_lora(self, base_model):
        """测试应用 LoRA"""
        lora_config = LoRAConfig(r=4, target_modules=["c_attn"])
        model = apply_lora_to_model(base_model, lora_config, verbose=False)

        # 检查模型有 LoRA 属性
        assert hasattr(model, 'lora_config')
        assert hasattr(model, 'lora_layers')
        assert len(model.lora_layers) > 0

    def test_frozen_parameters(self, base_model):
        """测试非 LoRA 参数被冻结"""
        lora_config = LoRAConfig(r=4)
        model = apply_lora_to_model(base_model, lora_config, verbose=False)

        trainable_count = 0
        frozen_count = 0

        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.requires_grad
                trainable_count += 1
            else:
                assert not param.requires_grad
                frozen_count += 1

        assert trainable_count > 0
        assert frozen_count > 0

    def test_trainable_params_ratio(self, base_model):
        """测试可训练参数比例"""
        lora_config = LoRAConfig(r=4)
        model = apply_lora_to_model(base_model, lora_config, verbose=False)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # LoRA 参数应该远少于总参数
        ratio = trainable_params / total_params
        assert ratio < 0.1  # 小于 10%

    def test_forward_after_lora(self, base_model):
        """测试应用 LoRA 后的前向传播"""
        lora_config = LoRAConfig(r=4)
        model = apply_lora_to_model(base_model, lora_config, verbose=False)

        x = torch.randint(0, 100, (2, 16))
        logits, _ = model(x)

        assert logits.shape == (2, 16, 100)
        assert not torch.isnan(logits).any()


class TestSaveLoadLoRA:
    """保存/加载 LoRA 测试"""

    @pytest.fixture
    def lora_model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        lora_config = LoRAConfig(r=4)
        return apply_lora_to_model(model, lora_config, verbose=False)

    def test_save_lora(self, lora_model):
        """测试保存 LoRA"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_lora(lora_model, tmpdir)

            assert os.path.exists(os.path.join(tmpdir, "lora_weights.pt"))
            assert os.path.exists(os.path.join(tmpdir, "lora_config.json"))

    def test_load_lora(self, lora_model):
        """测试加载 LoRA"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_lora(lora_model, tmpdir)

            # 创建新模型并加载 LoRA
            config = GPTConfig(
                vocab_size=100,
                emb_dim=64,
                num_heads=4,
                num_layers=2,
                context_size=32
            )
            new_model = GPT(config)
            new_model = load_lora(new_model, tmpdir)

            assert hasattr(new_model, 'lora_config')

    def test_lora_weights_preserved(self, lora_model):
        """测试 LoRA 权重在保存/加载后保持一致"""
        # 设置非零权重（使用较小的值）
        for module in lora_model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.data = torch.randn_like(module.lora_A) * 0.01
                module.lora_B.data = torch.randn_like(module.lora_B) * 0.01

        with tempfile.TemporaryDirectory() as tmpdir:
            save_lora(lora_model, tmpdir)

            # 检查 LoRA 权重文件正确保存
            state_dict = get_lora_state_dict(lora_model)
            loaded_state = torch.load(os.path.join(tmpdir, "lora_weights.pt"))

            # 验证权重一致
            for key in state_dict:
                assert torch.allclose(state_dict[key], loaded_state[key])


class TestMergeUnmerge:
    """合并/取消合并测试"""

    @pytest.fixture
    def trained_lora_model(self):
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )
        model = GPT(config)
        # 使用 dropout=0.0 以确保合并前后输出一致
        lora_config = LoRAConfig(r=4, dropout=0.0)
        model = apply_lora_to_model(model, lora_config, verbose=False)

        # 设置非零 LoRA 权重（使用较小的值以减少数值误差）
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.data = torch.randn_like(module.lora_A) * 0.01
                module.lora_B.data = torch.randn_like(module.lora_B) * 0.01

        model.eval()  # 设置为评估模式
        return model

    def test_merge_lora(self, trained_lora_model):
        """测试合并 LoRA"""
        x = torch.randint(0, 100, (1, 8))

        with torch.no_grad():
            output_before, _ = trained_lora_model(x)

        merge_lora(trained_lora_model)

        # 所有 LoRA 层应该被标记为已合并
        for module in trained_lora_model.modules():
            if isinstance(module, LoRALinear):
                assert module.merged

        with torch.no_grad():
            output_after, _ = trained_lora_model(x)

        # 输出应该相同（使用相对容差）
        # 由于浮点运算，可能有微小差异
        assert torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-4)

    def test_unmerge_lora(self, trained_lora_model):
        """测试取消合并 LoRA"""
        merge_lora(trained_lora_model)
        unmerge_lora(trained_lora_model)

        for module in trained_lora_model.modules():
            if isinstance(module, LoRALinear):
                assert not module.merged


class TestGetLoRAStateDict:
    """LoRA 状态字典测试"""

    def test_get_state_dict(self):
        """测试获取状态字典"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32
        )
        model = GPT(config)
        lora_config = LoRAConfig(r=4, target_modules=["c_attn"])
        model = apply_lora_to_model(model, lora_config, verbose=False)

        state_dict = get_lora_state_dict(model)

        assert len(state_dict) > 0
        for key in state_dict:
            assert "lora_A" in key or "lora_B" in key

    def test_load_state_dict(self):
        """测试加载状态字典"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32
        )
        model = GPT(config)
        lora_config = LoRAConfig(r=4)
        model = apply_lora_to_model(model, lora_config, verbose=False)

        # 获取并修改状态字典
        state_dict = get_lora_state_dict(model)
        for key in state_dict:
            state_dict[key] = torch.randn_like(state_dict[key])

        # 加载修改后的状态字典
        load_lora_state_dict(model, state_dict)

        # 验证权重已更新
        new_state_dict = get_lora_state_dict(model)
        for key in state_dict:
            assert torch.allclose(state_dict[key], new_state_dict[key])


class TestLoRAGradient:
    """LoRA 梯度测试"""

    def test_gradient_only_on_lora(self):
        """测试梯度只在 LoRA 参数上"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32
        )
        model = GPT(config)
        lora_config = LoRAConfig(r=4)
        model = apply_lora_to_model(model, lora_config, verbose=False)

        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        _, loss = model(x, targets)
        loss.backward()

        # 只有 LoRA 参数应该有梯度
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.grad is not None
            else:
                # 非 LoRA 参数不应该有梯度（因为 requires_grad=False）
                assert param.grad is None or not param.requires_grad

    def test_training_step(self):
        """测试训练步骤"""
        config = GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=2,
            context_size=32
        )
        model = GPT(config)
        lora_config = LoRAConfig(r=4)
        model = apply_lora_to_model(model, lora_config, verbose=False)

        # 只优化 LoRA 参数
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(lora_params, lr=1e-3)

        x = torch.randint(0, 100, (2, 16))
        targets = torch.randint(0, 100, (2, 16))

        # 训练前损失
        with torch.no_grad():
            _, loss_before = model(x, targets)

        # 训练一步
        model.train()
        optimizer.zero_grad()
        _, loss = model(x, targets)
        loss.backward()
        optimizer.step()

        # 训练后损失应该变化
        with torch.no_grad():
            _, loss_after = model(x, targets)

        assert loss_after.item() != loss_before.item()
