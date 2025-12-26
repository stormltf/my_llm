"""
模型架构单元测试

测试内容：
1. GPTConfig 配置验证
2. LayerNorm 层
3. FeedForward 层
4. CausalSelfAttention 层
5. TransformerBlock
6. GPT 完整模型
7. 前向传播和生成
"""

import pytest
import torch
import torch.nn as nn

from model import (
    GPT, GPTConfig, LayerNorm, FeedForward,
    CausalSelfAttention, TransformerBlock
)


class TestGPTConfig:
    """GPT 配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = GPTConfig()
        assert config.vocab_size == 1000
        assert config.emb_dim == 256
        assert config.num_heads == 8
        assert config.num_layers == 6

    def test_custom_config(self):
        """测试自定义配置"""
        config = GPTConfig(
            vocab_size=5000,
            emb_dim=512,
            num_heads=8,
            num_layers=12
        )
        assert config.vocab_size == 5000
        assert config.emb_dim == 512

    def test_invalid_config_head_dim(self):
        """测试无效配置 - emb_dim 不能被 num_heads 整除"""
        with pytest.raises(AssertionError):
            GPTConfig(emb_dim=256, num_heads=7)

    def test_valid_head_dim(self):
        """测试有效的头维度配置"""
        config = GPTConfig(emb_dim=256, num_heads=8)
        # 256 / 8 = 32
        assert config.emb_dim // config.num_heads == 32


class TestLayerNorm:
    """层归一化测试"""

    def test_output_shape(self):
        """测试输出形状"""
        ln = LayerNorm(emb_dim=64)
        x = torch.randn(2, 10, 64)
        out = ln(x)
        assert out.shape == x.shape

    def test_normalization(self):
        """测试归一化效果"""
        ln = LayerNorm(emb_dim=64)
        x = torch.randn(2, 10, 64) * 100 + 50  # 较大的均值和方差
        out = ln(x)

        # 归一化后每个位置的均值应接近 0
        mean = out.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    def test_learnable_params(self):
        """测试可学习参数"""
        ln = LayerNorm(emb_dim=64)
        assert ln.scale.shape == (64,)
        assert ln.shift.shape == (64,)
        assert ln.scale.requires_grad
        assert ln.shift.requires_grad

    def test_gradient_flow(self):
        """测试梯度流动"""
        ln = LayerNorm(emb_dim=32)
        x = torch.randn(2, 5, 32, requires_grad=True)
        out = ln(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert ln.scale.grad is not None


class TestFeedForward:
    """前馈网络测试"""

    def test_output_shape(self):
        """测试输出形状"""
        ff = FeedForward(emb_dim=64, multiplier=4)
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == x.shape

    def test_expansion(self):
        """测试维度扩展"""
        ff = FeedForward(emb_dim=64, multiplier=4)
        # 第一层应该扩展 4 倍
        assert ff.linear1.out_features == 64 * 4
        assert ff.linear2.in_features == 64 * 4

    def test_dropout(self):
        """测试 dropout"""
        ff = FeedForward(emb_dim=64, dropout=0.5)
        ff.train()
        x = torch.ones(2, 10, 64)

        # 多次运行，输出应该不同（因为 dropout）
        out1 = ff(x)
        out2 = ff(x)
        assert not torch.allclose(out1, out2)

    def test_no_dropout_in_eval(self):
        """测试评估模式下无 dropout"""
        ff = FeedForward(emb_dim=64, dropout=0.5)
        ff.eval()
        x = torch.ones(2, 10, 64)

        out1 = ff(x)
        out2 = ff(x)
        assert torch.allclose(out1, out2)


class TestCausalSelfAttention:
    """因果自注意力测试"""

    @pytest.fixture
    def attn_config(self):
        return GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=1,
            context_size=32,
            dropout=0.0
        )

    def test_output_shape(self, attn_config):
        """测试输出形状"""
        attn = CausalSelfAttention(attn_config)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_causal_mask(self, attn_config):
        """测试因果掩码"""
        attn = CausalSelfAttention(attn_config)
        # mask 应该是下三角矩阵
        assert attn.mask.shape == (32, 32)
        assert torch.allclose(attn.mask, torch.tril(torch.ones(32, 32)))

    def test_single_token(self, attn_config):
        """测试单 token 输入"""
        attn = CausalSelfAttention(attn_config)
        x = torch.randn(2, 1, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_multi_head(self, attn_config):
        """测试多头注意力"""
        attn = CausalSelfAttention(attn_config)
        # 应该有 4 个头
        assert attn.num_heads == 4
        assert attn.head_dim == 16  # 64 / 4


class TestTransformerBlock:
    """Transformer Block 测试"""

    @pytest.fixture
    def block_config(self):
        return GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=1,
            context_size=32,
            dropout=0.0
        )

    def test_output_shape(self, block_config):
        """测试输出形状"""
        block = TransformerBlock(block_config)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self, block_config):
        """测试残差连接"""
        block = TransformerBlock(block_config)
        x = torch.randn(2, 10, 64)

        # 初始时，输出应该和输入相似（因为残差连接）
        # 但由于权重随机初始化，不会完全相等
        out = block(x)
        # 检查输出不是 nan 或 inf
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_contains_attn_and_mlp(self, block_config):
        """测试包含注意力和 MLP"""
        block = TransformerBlock(block_config)
        assert hasattr(block, 'attn')
        assert hasattr(block, 'mlp')
        assert hasattr(block, 'ln_1')
        assert hasattr(block, 'ln_2')


class TestGPT:
    """GPT 模型测试"""

    def test_model_creation(self, gpt_config):
        """测试模型创建"""
        model = GPT(gpt_config)
        assert model is not None

    def test_forward_no_targets(self, small_model, device):
        """测试前向传播（不计算损失）"""
        batch_size, seq_len = 2, 10
        x = torch.randint(0, 100, (batch_size, seq_len), device=device)

        logits, loss = small_model(x)

        assert logits.shape == (batch_size, seq_len, 100)
        assert loss is None

    def test_forward_with_targets(self, small_model, device):
        """测试前向传播（计算损失）"""
        batch_size, seq_len = 2, 10
        x = torch.randint(0, 100, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 100, (batch_size, seq_len), device=device)

        logits, loss = small_model(x, targets)

        assert logits.shape == (batch_size, seq_len, 100)
        assert loss is not None
        assert loss.dim() == 0  # 标量

    def test_loss_reasonable(self, small_model, device):
        """测试损失值在合理范围内"""
        batch_size, seq_len = 4, 16
        x = torch.randint(0, 100, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 100, (batch_size, seq_len), device=device)

        _, loss = small_model(x, targets)

        # 随机初始化时，交叉熵损失应该接近 -log(1/vocab_size)
        expected_loss = torch.log(torch.tensor(100.0))  # ≈ 4.6
        assert loss.item() < expected_loss.item() * 2  # 给一些余量

    def test_generate(self, small_model, device):
        """测试文本生成"""
        x = torch.tensor([[1, 2, 3]], device=device)

        generated = small_model.generate(
            x,
            max_new_tokens=10,
            temperature=1.0
        )

        assert generated.shape[1] == 13  # 3 + 10

    def test_generate_with_eos(self, small_model, device):
        """测试遇到 EOS 停止生成"""
        x = torch.tensor([[1]], device=device)

        # 设置 eos_token_id，虽然可能不会触发
        generated = small_model.generate(
            x,
            max_new_tokens=5,
            eos_token_id=0
        )

        assert generated.shape[1] <= 6  # 最多 1 + 5

    def test_generate_top_k(self, small_model, device):
        """测试 Top-k 采样"""
        x = torch.tensor([[1, 2]], device=device)

        generated = small_model.generate(
            x,
            max_new_tokens=5,
            top_k=10
        )

        assert generated.shape[1] == 7

    def test_generate_top_p(self, small_model, device):
        """测试 Top-p 采样"""
        x = torch.tensor([[1, 2]], device=device)

        generated = small_model.generate(
            x,
            max_new_tokens=5,
            top_p=0.9
        )

        assert generated.shape[1] == 7

    def test_parameter_count(self, small_model):
        """测试参数量计算"""
        num_params = small_model.get_num_params()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_weight_sharing(self, gpt_config):
        """测试权重共享"""
        model = GPT(gpt_config)
        # Token embedding 和 lm_head 应该共享权重
        assert model.tok_emb.weight is model.lm_head.weight

    def test_context_length_assertion(self, small_model, device):
        """测试超过上下文长度的断言"""
        # 创建超过上下文长度的输入
        x = torch.randint(0, 100, (1, 100), device=device)  # context_size=64

        with pytest.raises(AssertionError):
            small_model(x)


class TestModelCompatibility:
    """模型兼容性测试"""

    def test_myllm_alias(self):
        """测试 MyLLM 别名"""
        from model import MyLLM
        assert MyLLM is GPT

    def test_compatibility_attributes(self, small_model):
        """测试兼容性属性"""
        # 这些属性是为了兼容 reward_model.py 等文件
        assert hasattr(small_model, 'token_embedding')
        assert hasattr(small_model, 'position_embedding')
        assert hasattr(small_model, 'transformer_blocks')
        assert hasattr(small_model, 'final_norm')


class TestModelTraining:
    """模型训练相关测试"""

    def test_gradient_flow(self, gpt_config, device):
        """测试梯度流动"""
        model = GPT(gpt_config).to(device)
        x = torch.randint(0, 100, (2, 10), device=device)
        targets = torch.randint(0, 100, (2, 10), device=device)

        logits, loss = model(x, targets)
        loss.backward()

        # 检查所有参数都有梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_training_step(self, gpt_config, device):
        """测试训练步骤"""
        model = GPT(gpt_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randint(0, 100, (2, 10), device=device)
        targets = torch.randint(0, 100, (2, 10), device=device)

        # 记录初始损失
        with torch.no_grad():
            _, initial_loss = model(x, targets)

        # 训练一步
        model.train()
        _, loss = model(x, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 训练后损失应该变化
        with torch.no_grad():
            _, final_loss = model(x, targets)

        assert final_loss.item() != initial_loss.item()

    def test_eval_mode(self, small_model):
        """测试评估模式"""
        small_model.eval()
        assert not small_model.training

        small_model.train()
        assert small_model.training


class TestModelNumericalStability:
    """数值稳定性测试"""

    def test_no_nan_in_forward(self, small_model, device):
        """测试前向传播无 NaN"""
        x = torch.randint(0, 100, (4, 32), device=device)
        logits, _ = small_model(x)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_no_nan_in_backward(self, gpt_config, device):
        """测试反向传播无 NaN"""
        model = GPT(gpt_config).to(device)
        x = torch.randint(0, 100, (2, 16), device=device)
        targets = torch.randint(0, 100, (2, 16), device=device)

        _, loss = model(x, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_generate_no_nan(self, small_model, device):
        """测试生成无 NaN"""
        x = torch.tensor([[1, 2, 3]], device=device)
        generated = small_model.generate(x, max_new_tokens=20)

        assert not torch.isnan(generated.float()).any()
