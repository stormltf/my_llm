"""
大语言模型核心组件实现

包含：
1. LayerNorm: 层归一化
2. FeedForward: 前馈神经网络
3. Attention: 注意力机制
4. MultiHeadAttention: 多头注意力
5. TransformerBlock: Transformer 块
6. GPT: 完整的 GPT 模型

参考：GPT-2, LLaMA 架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    """
    GPT 模型配置

    这里定义了模型的所有超参数，相当于"设计图纸"
    """
    # 模型结构参数
    vocab_size: int = 1000          # 词表大小（需要和 tokenizer 匹配）
    emb_dim: int = 256              # 词嵌入维度（每个token用多少维向量表示）
    num_heads: int = 8              # 多头注意力的头数（需要能被 emb_dim 整除）
    num_layers: int = 6             # Transformer Block 的层数
    context_size: int = 256         # 最大上下文长度（一次最多看多少个token）

    # 正则化参数
    dropout: float = 0.1            # Dropout 比例（防止过拟合）
    layer_norm_epsilon: float = 1e-5

    # 前馈网络参数
    ffn_multiplier: int = 4         # FFN 中间层维度放大倍数

    def __post_init__(self):
        """配置验证"""
        assert self.emb_dim % self.num_heads == 0, \
            f"emb_dim ({self.emb_dim}) 必须能被 num_heads ({self.num_heads}) 整除"


class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)

    作用：将每一层的输入归一化，使数值分布更稳定
    公式：LN(x) = scale * (x - mean) / sqrt(var + eps) + shift

    对比 BatchNorm：
    - BatchNorm: 对 batch 维度做归一化（CNN 中常用）
    - LayerNorm: 对特征维度做归一化（NLP 中常用，因为序列长度不固定）
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习参数：缩放和平移
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 [batch_size, seq_len, emb_dim]

        Returns:
            归一化后的张量
        """
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(dim=-1, keepdim=True)      # [batch, seq_len, 1]
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # [batch, seq_len, 1]

        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 应用可学习的缩放和平移
        return self.scale * x_norm + self.shift


class FeedForward(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)

    结构：Linear -> GELU -> Linear
    作用：在 Attention 之后进行"思考"和"记忆"

    维度变化：emb_dim -> emb_dim * multiplier -> emb_dim
    例如：256 -> 1024 -> 256
    """

    def __init__(self, emb_dim: int, multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        # 第一层：维度放大（增加模型容量，捕捉更多模式）
        self.linear1 = nn.Linear(emb_dim, emb_dim * multiplier)
        # 第二层：维度还原
        self.linear2 = nn.Linear(emb_dim * multiplier, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 [batch_size, seq_len, emb_dim]

        Returns:
            输出张量，形状 [batch_size, seq_len, emb_dim]
        """
        # 维度放大 + 激活函数
        x = self.linear1(x)
        x = F.gelu(x)  # GELU 是目前 LLM 最常用的激活函数
        # 维度还原
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制 (Causal Self-Attention)

    核心思想：
    1. 每个 token 作为 Query 去查询其他所有 token
    2. 通过 Q @ K 计算相似度（注意力分数）
    3. 用注意力分数对 Value 进行加权求和
    4. Causal（因果）：当前 token 只能看到之前的 token（用 Mask 实现）

    数学公式：
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.num_heads = config.num_heads
        self.head_dim = config.emb_dim // config.num_heads  # 每个头负责的维度
        self.context_size = config.context_size

        # 确保 emb_dim 能被 num_heads 整除
        assert self.head_dim * self.num_heads == self.emb_dim, \
            f"emb_dim ({config.emb_dim}) 必须能被 num_heads ({config.num_heads}) 整除"

        # Q, K, V 的线性变换层
        # 注意：这里不分开定义三个矩阵，而是一个大矩阵，提高并行效率
        self.c_attn = nn.Linear(config.emb_dim, 3 * config.emb_dim)  # 输出 Q, K, V 拼接

        # 输出投影层
        self.c_proj = nn.Linear(config.emb_dim, config.emb_dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果 Mask（下三角矩阵）
        # 作用：让当前位置只能看到之前的位置，看不到未来的位置
        # 例如：位置 2 可以看到位置 0, 1, 2，但看不到 3, 4, 5...
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_size, config.context_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 [batch_size, seq_len, emb_dim]

        Returns:
            注意力输出，形状 [batch_size, seq_len, emb_dim]
        """
        batch_size, seq_len, emb_dim = x.shape

        # 1. 计算 Q, K, V
        # qkv 形状: [batch, seq_len, 3 * emb_dim]
        qkv = self.c_attn(x)

        # 分割成 Q, K, V
        # 每个形状: [batch, seq_len, emb_dim]
        q, k, v = qkv.split(self.emb_dim, dim=2)

        # 2. 重塑为多头形式
        # 形状变换: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 3. 转置，方便并行计算
        # 形状: [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. 计算注意力分数 (Q @ K^T)
        # 形状: [batch, num_heads, seq_len, seq_len]
        # attn_score[i, h, j, k] 表示第 i 个样本、第 h 个头中，位置 j 对位置 k 的注意力分数
        attn_score = q @ k.transpose(-2, -1)

        # 5. 缩放 (除以 sqrt(d_k))，防止数值过大导致梯度消失
        attn_score = attn_score / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # 6. 应用因果 Mask（把未来位置的分数设为 -inf）
        # 这样 softmax 后这些位置的权重就会接近 0
        if seq_len > 1:  # 推理时 seq_len=1 不需要 mask
            # 只取需要的 mask 部分
            mask = self.mask[:seq_len, :seq_len]
            # 将 mask=0 的位置设为 -inf
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        # 7. Softmax 归一化，得到注意力权重
        # 形状: [batch, num_heads, seq_len, seq_len]
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        # 8. 加权求和 (权重 @ Value)
        # 形状: [batch, num_heads, seq_len, head_dim]
        context_vec = attn_weight @ v

        # 9. 合并多头
        # 先转回: [batch, seq_len, num_heads, head_dim]
        context_vec = context_vec.transpose(1, 2)
        # 再 reshape: [batch, seq_len, emb_dim]
        context_vec = context_vec.reshape(batch_size, seq_len, emb_dim)

        # 10. 输出投影
        output = self.c_proj(context_vec)
        output = self.resid_dropout(output)

        return output


class TransformerBlock(nn.Module):
    """
    Transformer Block (Transformer 块)

    这是 GPT 的基本构建单元，包含：
    1. Layer Norm -> Multi-Head Attention -> Residual
    2. Layer Norm -> Feed Forward -> Residual

    注意：这里使用 Pre-Norm（Norm 在前面），这是目前主流做法（LLaMA, GPT-2, Qwen 等）
    相比 Post-Norm（Norm 在后面），Pre-Norm 训练更稳定。
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # 第一个子层：自注意力
        self.ln_1 = LayerNorm(config.emb_dim, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)

        # 第二个子层：前馈网络
        self.ln_2 = LayerNorm(config.emb_dim, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(
            config.emb_dim,
            multiplier=config.ffn_multiplier,
            dropout=config.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 [batch_size, seq_len, emb_dim]

        Returns:
            输出张量，形状 [batch_size, seq_len, emb_dim]
        """
        # 子层1：Attention + Residual
        # 注意：Residual 是 x + layer_output，保证梯度不会消失
        x = x + self.attn(self.ln_1(x))

        # 子层2：FeedForward + Residual
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) 模型

    完整的大语言模型结构：
    Token Embedding -> Position Embedding -> [Transformer Block] * N -> Layer Norm -> Output Projection
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 1. Token Embedding：将 token ID 映射为向量
        # 形状: [vocab_size, emb_dim]
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)

        # 2. Position Embedding：为每个位置分配一个向量
        # 形状: [context_size, emb_dim]
        self.pos_emb = nn.Embedding(config.context_size, config.emb_dim)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # 3. Transformer Blocks 堆叠
        # 这是模型的"大脑"部分
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # 4. 最后的 Layer Norm
        self.ln_f = LayerNorm(config.emb_dim, eps=config.layer_norm_epsilon)

        # 5. 输出投影：将向量映射回词表
        # 形状: [emb_dim, vocab_size]
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # 权重共享：Token Embedding 和输出层共享权重（可以减少参数量）
        # 这是 GPT-2 开始的做法
        self.tok_emb.weight = self.lm_head.weight

        # 初始化权重（重要！好的初始化能让训练更稳定）
        self.apply(self._init_weights)

        # 兼容性别名（供 reward_model.py 使用）
        self.token_embedding = self.tok_emb
        self.position_embedding = self.pos_emb
        self.transformer_blocks = self.blocks
        self.final_norm = self.ln_f

        print(f"GPT 模型初始化完成，参数量: {self.get_num_params():,}")

    def _init_weights(self, module):
        """
        权重初始化

        参考 GPT-2 的初始化方案：
        - Linear: 正态分布，std=0.02
        - Embedding: 正态分布，std=0.02
        - LayerNorm: scale=1, shift=0
        """
        if isinstance(module, nn.Linear):
            # 特殊处理最后一层 lm_head，让输出方差小一些
            if module is self.lm_head:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            torch.nn.init.ones_(module.scale)
            torch.nn.init.zeros_(module.shift)

    def get_num_params(self):
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播

        Args:
            idx: 输入 token ID 序列，形状 [batch_size, seq_len]
            targets: 目标 token ID 序列（训练时使用），形状 [batch_size, seq_len]

        Returns:
            logits: 模型输出，形状 [batch_size, seq_len, vocab_size]
            loss: 损失值（只有训练时才计算）
        """
        device = idx.device
        batch_size, seq_len = idx.shape

        # 确保 seq_len 不超过最大上下文长度
        assert seq_len <= self.config.context_size, \
            f"序列长度 {seq_len} 超过最大上下文长度 {self.config.context_size}"

        # 1. 获取 token embeddings
        # 形状: [batch_size, seq_len, emb_dim]
        tok_emb = self.tok_emb(idx)

        # 2. 获取 position embeddings
        # 生成位置索引: [0, 1, 2, ..., seq_len-1]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        # 形状: [seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
        pos_emb = self.pos_emb(pos).unsqueeze(0)

        # 3. 合并 token 和 position embeddings
        # 形状: [batch_size, seq_len, emb_dim]
        x = self.drop(tok_emb + pos_emb)

        # 4. 通过 Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 5. 最后的 Layer Norm
        x = self.ln_f(x)

        # 6. 输出投影
        # 形状: [batch_size, seq_len, vocab_size]
        logits = self.lm_head(x)

        # 7. 计算损失（如果提供了 targets）
        loss = None
        if targets is not None:
            # 将 logits 展平为 [batch_size * seq_len, vocab_size]
            # 将 targets 展平为 [batch_size * seq_len]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # 忽略 padding 位置的 loss
            )

        return logits, loss

    def num_parameters(self) -> int:
        """返回模型参数量（兼容接口）"""
        return self.get_num_params()

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int = None
    ) -> torch.Tensor:
        """
        自回归文本生成

        Args:
            idx: 输入 token ID 序列，形状 [batch_size, seq_len]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数，控制随机性
            top_k: 只从概率最高的 k 个 token 中采样
            top_p: 只从累积概率达到 p 的 token 中采样
            eos_token_id: 结束 token ID，遇到时停止生成

        Returns:
            生成的完整序列，形状 [batch_size, seq_len + new_tokens]
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 截断输入，确保不超过上下文长度
            idx_cond = idx if idx.size(1) <= self.config.context_size else idx[:, -self.config.context_size:]

            # 前向传播
            logits, _ = self(idx_cond)

            # 取最后一个位置的 logits
            logits = logits[:, -1, :] / temperature

            # Top-k 采样
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接到序列
            idx = torch.cat((idx, idx_next), dim=1)

            # 检查是否遇到结束 token
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return idx


# ==========================================
# 兼容性别名
# ==========================================
# 为了兼容 reward_model.py, rlhf.py, rlvf.py 中的导入
MyLLM = GPT


def demo_model():
    """
    模型使用示例
    """
    print("=" * 60)
    print("GPT 模型示例")
    print("=" * 60)

    # 创建配置
    config = GPTConfig(
        vocab_size=1000,
        emb_dim=256,
        num_heads=8,
        num_layers=6,
        context_size=256,
        dropout=0.1
    )

    # 创建模型
    model = GPT(config)

    # 测试前向传播
    batch_size = 2
    seq_len = 10
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\n输入形状: {idx.shape}")

    # 前向传播（不计算loss）
    logits, loss = model(idx)
    print(f"输出形状: {logits.shape}")
    print(f"Loss: {loss}")

    # 前向传播（计算loss）
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(idx, targets)
    print(f"\n带训练的输出形状: {logits.shape}")
    print(f"训练Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)


def demo_forward_step_by_step():
    """
    🎓 前向传播详细演示

    用一个超小模型，逐步展示数据如何流过网络的每一层。
    这个演示使用具体数字，让你直观理解前向传播的全过程。
    """
    print("\n" + "=" * 70)
    print("🎓 前向传播详细演示 - 用具体数字理解每一步")
    print("=" * 70)

    # ================================================================
    # 配置一个超小模型，便于观察
    # ================================================================
    print("\n📋 【第0步：模型配置】")
    print("-" * 50)

    config = GPTConfig(
        vocab_size=5,      # 只有5个词：假设是 ["我", "爱", "AI", "学", "习"]
        emb_dim=4,         # 每个词用4个数字表示
        num_heads=2,       # 2个注意力头
        num_layers=1,      # 只用1层Transformer（便于观察）
        context_size=8,    # 最大看8个词
        dropout=0.0,       # 关闭dropout，结果可复现
    )

    print(f"   词表大小 (vocab_size):  {config.vocab_size} 个词")
    print(f"   嵌入维度 (emb_dim):     {config.emb_dim} 维向量")
    print(f"   注意力头数 (num_heads): {config.num_heads} 个头")
    print(f"   每个头的维度:           {config.emb_dim // config.num_heads} 维")
    print(f"   Transformer层数:        {config.num_layers} 层")

    # 设置随机种子，让结果可复现
    torch.manual_seed(42)

    # 创建模型（不打印参数量信息）
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    model = GPT(config)
    sys.stdout = old_stdout
    model.eval()  # 评估模式

    # ================================================================
    # 准备输入
    # ================================================================
    print("\n📥 【第1步：准备输入】")
    print("-" * 50)

    # 假设输入是 "我 爱" -> token ID = [0, 1]
    idx = torch.tensor([[0, 1]])  # 形状: [1, 2] (1个句子，2个词)

    print(f"   假设词表: ['我', '爱', 'AI', '学', '习']")
    print(f"            ID:  0     1     2     3     4")
    print(f"")
    print(f"   输入句子: '我 爱'")
    print(f"   Token IDs: {idx.tolist()[0]}")
    print(f"   张量形状: {list(idx.shape)} = [batch_size=1, seq_len=2]")

    # ================================================================
    # 第2步：Token Embedding
    # ================================================================
    print("\n📊 【第2步：Token Embedding - 把ID变成向量】")
    print("-" * 50)

    tok_emb = model.tok_emb(idx)

    print(f"   操作: tok_emb = model.tok_emb(idx)")
    print(f"   原理: 用 token ID 去嵌入表里查对应的行")
    print(f"")
    print(f"   嵌入表 (vocab_size × emb_dim = 5 × 4):")
    emb_weight = model.tok_emb.weight.data
    for i in range(config.vocab_size):
        word = ['我', '爱', 'AI', '学', '习'][i]
        vec = emb_weight[i].tolist()
        vec_str = [f"{v:+.3f}" for v in vec]
        print(f"      ID={i} '{word}' → [{', '.join(vec_str)}]")

    print(f"")
    print(f"   查表过程:")
    print(f"      ID=0 '我' → 取第0行")
    print(f"      ID=1 '爱' → 取第1行")
    print(f"")
    print(f"   tok_emb 结果 (形状 {list(tok_emb.shape)} = [batch, seq_len, emb_dim]):")
    for i, word in enumerate(['我', '爱']):
        vec = tok_emb[0, i].tolist()
        vec_str = [f"{v:+.3f}" for v in vec]
        print(f"      '{word}' → [{', '.join(vec_str)}]")

    # ================================================================
    # 第3步：Position Embedding
    # ================================================================
    print("\n📍 【第3步：Position Embedding - 加上位置信息】")
    print("-" * 50)

    seq_len = idx.shape[1]
    pos = torch.arange(0, seq_len, dtype=torch.long)
    pos_emb = model.pos_emb(pos)

    print(f"   操作: pos_emb = model.pos_emb([0, 1])")
    print(f"   原理: 每个位置有一个对应的向量，告诉模型词的顺序")
    print(f"")
    print(f"   位置嵌入表 (context_size × emb_dim = 8 × 4) 前2行:")
    pos_weight = model.pos_emb.weight.data
    for i in range(2):
        vec = pos_weight[i].tolist()
        vec_str = [f"{v:+.3f}" for v in vec]
        print(f"      位置{i} → [{', '.join(vec_str)}]")

    print(f"")
    print(f"   pos_emb 结果 (形状 {list(pos_emb.shape)}):")
    for i in range(2):
        vec = pos_emb[i].tolist()
        vec_str = [f"{v:+.3f}" for v in vec]
        print(f"      位置{i} → [{', '.join(vec_str)}]")

    # ================================================================
    # 第4步：合并嵌入
    # ================================================================
    print("\n➕ 【第4步：合并嵌入 - Token向量 + 位置向量】")
    print("-" * 50)

    x = tok_emb + pos_emb.unsqueeze(0)

    print(f"   操作: x = tok_emb + pos_emb")
    print(f"   原理: 逐元素相加，让每个词同时知道'自己是谁'和'在哪个位置'")
    print(f"")
    print(f"   计算过程:")
    for i, word in enumerate(['我', '爱']):
        tok_vec = tok_emb[0, i].tolist()
        pos_vec = pos_emb[i].tolist()
        result = x[0, i].tolist()
        print(f"      '{word}' (位置{i}):")
        print(f"         Token:    [{', '.join([f'{v:+.3f}' for v in tok_vec])}]")
        print(f"       + Position: [{', '.join([f'{v:+.3f}' for v in pos_vec])}]")
        print(f"       = 合并结果: [{', '.join([f'{v:+.3f}' for v in result])}]")

    print(f"")
    print(f"   x 形状: {list(x.shape)} = [batch=1, seq_len=2, emb_dim=4]")

    # ================================================================
    # 第5步：Transformer Block
    # ================================================================
    print("\n🔄 【第5步：Transformer Block - 词之间交流】")
    print("-" * 50)
    print(f"   每个Block包含两个子层:")
    print(f"   1. Self-Attention: 词与词之间交流信息")
    print(f"   2. Feed-Forward:   每个词独立进行深度处理")

    # 进入第一个block
    block = model.blocks[0]

    # 5.1 LayerNorm + Attention
    print(f"\n   📌 5.1 Self-Attention 子层")
    print(f"   " + "-" * 40)

    ln_out = block.ln_1(x)
    print(f"   a) LayerNorm: 标准化数值")
    print(f"      输入 x:     均值={x.mean().item():.3f}, 标准差={x.std().item():.3f}")
    print(f"      输出 ln_x:  均值={ln_out.mean().item():.3f}, 标准差={ln_out.std().item():.3f}")

    # 计算Q, K, V
    attn = block.attn
    qkv = attn.c_attn(ln_out)
    q, k, v = qkv.split(config.emb_dim, dim=2)

    print(f"\n   b) 生成 Q, K, V:")
    print(f"      Q (Query - 我要找什么):  形状 {list(q.shape)}")
    print(f"      K (Key - 我是什么):      形状 {list(k.shape)}")
    print(f"      V (Value - 我的内容):    形状 {list(v.shape)}")

    # 重塑为多头形式
    batch_size = 1
    q = q.view(batch_size, seq_len, config.num_heads, config.emb_dim // config.num_heads).transpose(1, 2)
    k = k.view(batch_size, seq_len, config.num_heads, config.emb_dim // config.num_heads).transpose(1, 2)
    v = v.view(batch_size, seq_len, config.num_heads, config.emb_dim // config.num_heads).transpose(1, 2)

    print(f"\n   c) 重塑为多头形式 (2个头，每头2维):")
    print(f"      Q 形状: {list(q.shape)} = [batch, num_heads, seq_len, head_dim]")

    # 计算注意力分数
    head_dim = config.emb_dim // config.num_heads
    attn_score = q @ k.transpose(-2, -1) / (head_dim ** 0.5)

    print(f"\n   d) 计算注意力分数 (Q @ K^T / √d):")
    print(f"      注意力分数形状: {list(attn_score.shape)} = [batch, heads, seq, seq]")
    print(f"      ")
    print(f"      每个值表示: '行位置' 对 '列位置' 的关注程度")
    print(f"      ")
    for h in range(config.num_heads):
        print(f"      头{h}的注意力分数:")
        print(f"              '我'     '爱'")
        for i, word in enumerate(['我', '爱']):
            scores = attn_score[0, h, i].tolist()
            print(f"         '{word}':  {scores[0]:+.3f}   {scores[1]:+.3f}")

    # 应用因果Mask
    mask = torch.tril(torch.ones(seq_len, seq_len))
    attn_score_masked = attn_score.masked_fill(mask == 0, float('-inf'))

    print(f"\n   e) 应用因果Mask (当前词只能看之前的词):")
    print(f"      ")
    print(f"      Mask矩阵:     应用后:")
    print(f"        '我' '爱'     ")
    print(f"      [[1,   0],      '我'只能看自己")
    print(f"       [1,   1]]      '爱'可以看'我'和自己")
    print(f"      ")
    print(f"      头0 mask后:")
    print(f"              '我'     '爱'")
    for i, word in enumerate(['我', '爱']):
        scores = attn_score_masked[0, 0, i].tolist()
        s0 = f"{scores[0]:+.3f}" if scores[0] != float('-inf') else "  -∞  "
        s1 = f"{scores[1]:+.3f}" if scores[1] != float('-inf') else "  -∞  "
        print(f"         '{word}':  {s0}   {s1}")

    # Softmax
    attn_weight = F.softmax(attn_score_masked, dim=-1)

    print(f"\n   f) Softmax归一化 (变成概率，每行加起来=1):")
    print(f"      ")
    print(f"      头0的注意力权重:")
    print(f"              '我'     '爱'      (每行和=1)")
    for i, word in enumerate(['我', '爱']):
        weights = attn_weight[0, 0, i].tolist()
        row_sum = sum(weights)
        print(f"         '{word}':  {weights[0]:.3f}    {weights[1]:.3f}     ({row_sum:.3f})")

    print(f"\n      💡 解读:")
    print(f"         '我' → 100%关注自己 (因为看不到'爱')")
    print(f"         '爱' → 部分关注'我'，部分关注自己")

    # 加权求和
    context = attn_weight @ v

    print(f"\n   g) 用注意力权重加权求和 Value:")
    print(f"      context = attn_weight @ V")
    print(f"      形状: {list(context.shape)}")
    print(f"")
    print(f"      '我'的输出 = 1.0 × V['我'] + 0.0 × V['爱'] = V['我']")
    print(f"      '爱'的输出 = w1 × V['我'] + w2 × V['爱']  (加权混合)")

    # 残差连接
    attn_out = block.attn(block.ln_1(x))
    x_after_attn = x + attn_out

    print(f"\n   h) 残差连接: x = x + attention_output")
    print(f"      原理: 保留原始信息，防止深层网络梯度消失")

    # 5.2 Feed-Forward
    print(f"\n   📌 5.2 Feed-Forward 子层")
    print(f"   " + "-" * 40)

    ln2_out = block.ln_2(x_after_attn)
    ffn_out = block.mlp(ln2_out)
    x_after_ffn = x_after_attn + ffn_out

    print(f"   a) LayerNorm")
    print(f"   b) Feed-Forward: 扩大→激活→缩小")
    print(f"      输入维度: {config.emb_dim}")
    print(f"      中间维度: {config.emb_dim * 4} (扩大4倍)")
    print(f"      输出维度: {config.emb_dim}")
    print(f"   c) 残差连接")
    print(f"")
    print(f"   Block输出形状: {list(x_after_ffn.shape)}")

    # ================================================================
    # 第6步：最后的LayerNorm
    # ================================================================
    print("\n📏 【第6步：最后的LayerNorm】")
    print("-" * 50)

    # 通过所有blocks
    x_final = x
    for block in model.blocks:
        x_final = block(x_final)
    x_normed = model.ln_f(x_final)

    print(f"   操作: x = LayerNorm(x)")
    print(f"   原理: 最后再标准化一次，确保输出稳定")
    print(f"   输出形状: {list(x_normed.shape)}")

    # ================================================================
    # 第7步：输出投影
    # ================================================================
    print("\n🎯 【第7步：输出投影 - 向量变成词表概率】")
    print("-" * 50)

    logits = model.lm_head(x_normed)

    print(f"   操作: logits = model.lm_head(x)")
    print(f"   原理: 把{config.emb_dim}维向量映射到{config.vocab_size}维（词表大小）")
    print(f"")
    print(f"   权重矩阵形状: [{config.emb_dim}, {config.vocab_size}]")
    print(f"   logits 形状:  {list(logits.shape)} = [batch, seq_len, vocab_size]")
    print(f"")
    print(f"   每个位置的输出 (5个词的分数):")
    for i, word in enumerate(['我', '爱']):
        scores = logits[0, i].tolist()
        print(f"      位置{i} '{word}' 预测下一个词的分数:")
        for j, w in enumerate(['我', '爱', 'AI', '学', '习']):
            print(f"         '{w}': {scores[j]:+.3f}")

    # ================================================================
    # 第8步：预测下一个词
    # ================================================================
    print("\n🔮 【第8步：预测下一个词】")
    print("-" * 50)

    # 取最后一个位置的logits
    last_logits = logits[0, -1, :]  # 形状: [vocab_size]

    # Softmax得到概率
    probs = F.softmax(last_logits, dim=-1)

    print(f"   输入: '我 爱'")
    print(f"   任务: 预测 '爱' 后面的词")
    print(f"")
    print(f"   取最后位置的logits，用Softmax变成概率:")
    print(f"")
    words = ['我', '爱', 'AI', '学', '习']
    for j, w in enumerate(words):
        print(f"      '{w}': logit={last_logits[j]:+.3f} → 概率={probs[j]:.1%}")

    # 预测
    pred_id = probs.argmax().item()
    pred_word = words[pred_id]

    print(f"")
    print(f"   ✅ 预测结果: '{pred_word}' (概率最高)")
    print(f"   完整输出: '我 爱 {pred_word}'")

    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "=" * 70)
    print("📚 前向传播总结")
    print("=" * 70)
    print("""
    输入 "我 爱" [0, 1]                    形状变化
          ↓
    ┌─────────────────────────────────┐
    │ 1. Token Embedding              │   [1,2] → [1,2,4]
    │    ID → 向量 (查表)              │
    └─────────────────────────────────┘
          ↓
    ┌─────────────────────────────────┐
    │ 2. Position Embedding           │   + [2,4]
    │    加上位置信息                  │
    └─────────────────────────────────┘
          ↓
    ┌─────────────────────────────────┐
    │ 3. Transformer Block            │   [1,2,4] → [1,2,4]
    │    - Self-Attention: 词间交流   │
    │    - Feed-Forward: 深度处理     │
    └─────────────────────────────────┘
          ↓
    ┌─────────────────────────────────┐
    │ 4. LayerNorm                    │   [1,2,4] → [1,2,4]
    │    标准化输出                    │
    └─────────────────────────────────┘
          ↓
    ┌─────────────────────────────────┐
    │ 5. 输出投影 (lm_head)           │   [1,2,4] → [1,2,5]
    │    向量 → 词表概率              │
    └─────────────────────────────────┘
          ↓
    Softmax → 预测 "AI" (概率最高的词)
    """)
    print("=" * 70)


if __name__ == "__main__":
    demo_model()
    print("\n" + "🎓" * 30 + "\n")
    demo_forward_step_by_step()
