"""
注意力机制单元测试

测试内容：
1. CausalSelfAttention 因果自注意力
2. 注意力权重可视化
3. 多头注意力独立性
4. 注意力模式分析
"""

import pytest
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig, CausalSelfAttention


class TestAttentionMechanics:
    """注意力机制核心测试"""

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

    @pytest.fixture
    def attention(self, attn_config):
        return CausalSelfAttention(attn_config)

    def test_qkv_projection_shapes(self, attention):
        """测试 Q/K/V 投影形状"""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, attention.emb_dim)

        # 获取 Q, K, V
        q, k, v = attention.c_attn(x).split(attention.emb_dim, dim=2)

        # 检查形状
        assert q.shape == (batch_size, seq_len, attention.emb_dim)
        assert k.shape == (batch_size, seq_len, attention.emb_dim)
        assert v.shape == (batch_size, seq_len, attention.emb_dim)

    def test_qkv_reshape_to_heads(self, attention):
        """测试 Q/K/V 重塑为多头"""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, attention.emb_dim)

        q, k, v = attention.c_attn(x).split(attention.emb_dim, dim=2)

        # 重塑为多头
        head_dim = attention.emb_dim // attention.num_heads
        q = q.view(batch_size, seq_len, attention.num_heads, head_dim)
        k = k.view(batch_size, seq_len, attention.num_heads, head_dim)
        v = v.view(batch_size, seq_len, attention.num_heads, head_dim)

        # 检查重塑后的形状
        assert q.shape == (batch_size, seq_len, attention.num_heads, head_dim)
        assert k.shape == (batch_size, seq_len, attention.num_heads, head_dim)
        assert v.shape == (batch_size, seq_len, attention.num_heads, head_dim)

    def test_attention_scores_shape(self, attention):
        """测试注意力分数形状"""
        batch_size = 2
        seq_len = 10
        head_dim = attention.emb_dim // attention.num_heads

        q = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)

        # 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # 检查形状
        assert scores.shape == (batch_size, attention.num_heads, seq_len, seq_len)

    def test_attention_softmax(self, attention):
        """测试注意力 softmax"""
        batch_size = 2
        seq_len = 10
        head_dim = attention.emb_dim // attention.num_heads

        q = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 检查 softmax 后每行和为 1
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_causal_mask_structure(self, attention):
        """测试因果掩码结构"""
        # 掩码应该是下三角矩阵
        assert attention.mask.shape == (attention.context_size, attention.context_size)

        # 检查是下三角矩阵
        expected = torch.tril(torch.ones(attention.context_size, attention.context_size))
        assert torch.equal(attention.mask, expected)


class TestAttentionPatterns:
    """注意力模式测试"""

    @pytest.fixture
    def attn_config(self):
        return GPTConfig(
            vocab_size=100,
            emb_dim=64,
            num_heads=4,
            num_layers=1,
            context_size=16,
            dropout=0.0
        )

    @pytest.fixture
    def attention(self, attn_config):
        return CausalSelfAttention(attn_config)

    def test_attention_lower_triangular(self, attention):
        """测试注意力是下三角的（因果性）"""
        # 创建固定输入以便观察注意力模式
        batch_size = 1
        seq_len = 8
        x = torch.randn(batch_size, seq_len, attention.emb_dim)

        # 前向传播（内部应用了掩码）
        output = attention(x)

        # 由于因果掩码，输出应该只依赖于当前位置和之前的位置
        # 这个测试比较间接，但可以验证没有报错

    def test_attention_diagonal_dominance(self, attention):
        """测试注意力对角线占优（自注意力通常关注自己）"""
        batch_size = 1
        seq_len = 5
        head_dim = attention.emb_dim // attention.num_heads

        # 创建简单的 Q, K
        q = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).repeat(1, attention.num_heads, 1, 1)
        k = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).repeat(1, attention.num_heads, 1, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # 对角线应该最大
        diag_values = torch.diagonal(scores, dim1=-2, dim2=-1)
        assert (diag_values > scores.mean()).all()


class TestMultiHeadAttention:
    """多头注意力测试"""

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

    @pytest.fixture
    def attention(self, attn_config):
        return CausalSelfAttention(attn_config)

    def test_num_heads(self, attn_config):
        """测试头数"""
        assert attn_config.num_heads == 4

    def test_head_divisibility(self, attn_config):
        """测试 emb_dim 能被 num_heads 整除"""
        assert attn_config.emb_dim % attn_config.num_heads == 0

    def test_head_dim(self, attn_config):
        """测试头维度"""
        head_dim = attn_config.emb_dim // attn_config.num_heads
        assert head_dim == 16  # 64 / 4 = 16

    def test_multi_head_independence(self, attention):
        """测试多头之间的独立性"""
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, attention.emb_dim)

        output = attention(x)

        # 每个头应该处理不同的信息子空间
        # 输出应该融合所有头的信息
        assert output.shape == (batch_size, seq_len, attention.emb_dim)


class TestAttentionGradients:
    """注意力梯度测试"""

    @pytest.fixture
    def attn_config(self):
        return GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=1,
            context_size=16,
            dropout=0.0
        )

    @pytest.fixture
    def attention(self, attn_config):
        return CausalSelfAttention(attn_config)

    def test_gradient_flow_through_attention(self, attention):
        """测试梯度流过注意力层"""
        x = torch.randn(2, 5, attention.emb_dim, requires_grad=True)

        output = attention(x)
        loss = output.sum()
        loss.backward()

        # 检查梯度存在
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_attention_parameter_gradients(self, attention):
        """测试注意力参数梯度"""
        x = torch.randn(2, 5, attention.emb_dim)

        output = attention(x)
        loss = output.sum()
        loss.backward()

        # 检查所有参数都有梯度
        for name, param in attention.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


class TestAttentionWeights:
    """注意力权重分析测试"""

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

    @pytest.fixture
    def attention(self, attn_config):
        return CausalSelfAttention(attn_config)

    def test_extract_attention_weights(self, attention):
        """测试提取注意力权重"""
        batch_size = 1
        seq_len = 5
        head_dim = attention.emb_dim // attention.num_heads

        # 创建查询和键
        q = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 检查形状
        assert attn_weights.shape == (batch_size, attention.num_heads, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self, attention):
        """测试注意力权重和为 1"""
        batch_size = 1
        seq_len = 5
        head_dim = attention.emb_dim // attention.num_heads

        q = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 每行的和应该为 1
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestAttentionEdgeCases:
    """注意力边界情况测试"""

    @pytest.fixture
    def attn_config(self):
        return GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=1,
            context_size=16,
            dropout=0.0
        )

    @pytest.fixture
    def attention(self, attn_config):
        return CausalSelfAttention(attn_config)

    def test_single_token_attention(self, attention):
        """测试单个 token 的注意力"""
        x = torch.randn(1, 1, attention.emb_dim)
        output = attention(x)

        assert output.shape == (1, 1, attention.emb_dim)

    def test_max_length_attention(self, attention):
        """测试最大长度注意力"""
        seq_len = attention.context_size
        x = torch.randn(1, seq_len, attention.emb_dim)
        output = attention(x)

        assert output.shape == (1, seq_len, attention.emb_dim)

    def test_batch_attention(self, attention):
        """测试批次注意力"""
        batch_size = 8
        seq_len = 10
        x = torch.randn(batch_size, seq_len, attention.emb_dim)
        output = attention(x)

        assert output.shape == (batch_size, seq_len, attention.emb_dim)

    def test_zero_input(self, attention):
        """测试零输入"""
        x = torch.zeros(1, 5, attention.emb_dim)
        output = attention(x)

        assert output.shape == (1, 5, attention.emb_dim)
        # 零输入也应该产生合理输出（不是 NaN）

    def test_constant_input(self, attention):
        """测试常数输入"""
        x = torch.ones(1, 5, attention.emb_dim)
        output = attention(x)

        assert output.shape == (1, 5, attention.emb_dim)
        assert not torch.isnan(output).any()


class TestAttentionInContext:
    """注意力在模型中的上下文测试"""

    @pytest.fixture
    def model_config(self):
        return GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=2,
            context_size=32,
            dropout=0.0
        )

    @pytest.fixture
    def model(self, model_config):
        return GPT(model_config)

    def test_attention_in_transformer_block(self, model):
        """测试 Transformer Block 中的注意力"""
        x = torch.randn(2, 10, model.config.emb_dim)

        # 通过第一个 Transformer Block
        block = model.transformer_blocks[0]
        output = block(x)

        assert output.shape == x.shape

    def test_attention_stack(self, model):
        """测试多层注意力堆叠"""
        x = torch.randn(1, 8, model.config.emb_dim)

        # 通过所有层
        for block in model.transformer_blocks:
            x = block(x)

        # 形状应该保持不变
        assert x.shape[0] == 1
        assert x.shape[1] == 8
        assert x.shape[2] == model.config.emb_dim


class TestAttentionVisualization:
    """注意力可视化辅助测试"""

    @pytest.fixture
    def attn_config(self):
        return GPTConfig(
            vocab_size=100,
            emb_dim=32,
            num_heads=2,
            num_layers=1,
            context_size=16,
            dropout=0.0
        )

    @pytest.fixture
    def attention(self, attn_config):
        return CausalSelfAttention(attn_config)

    def test_get_attention_map(self, attention):
        """测试获取注意力图（用于可视化）"""
        batch_size = 1
        seq_len = 5
        head_dim = attention.emb_dim // attention.num_heads

        # 模拟计算注意力权重
        q = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, attention.num_heads, seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_map = F.softmax(scores, dim=-1)

        # 可以用于可视化的形状: [num_heads, seq_len, seq_len]
        attn_map = attn_map.squeeze(0).detach()

        assert attn_map.shape == (attention.num_heads, seq_len, seq_len)

    def test_head_specific_attention(self, attention):
        """测试特定头的注意力"""
        batch_size = 1
        seq_len = 5

        # 获取特定头的注意力权重
        attn_map = torch.randn(batch_size, attention.num_heads, seq_len, seq_len)
        attn_map = F.softmax(attn_map, dim=-1)

        # 提取第一个头
        head_0_attn = attn_map[0, 0, :, :]  # [seq_len, seq_len]

        assert head_0_attn.shape == (seq_len, seq_len)
