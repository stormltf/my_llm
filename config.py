"""
MyLLM 配置文件
======================
这个文件定义了模型的所有超参数配置。
通过修改这些参数，你可以调整模型的大小、容量和训练行为。

作者：MyLLM Team
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MyLLMConfig:
    """
    MyLLM 模型配置类

    这个类包含了构建和训练LLM所需的所有超参数。
    使用 dataclass 装饰器可以自动生成 __init__ 等方法。

    参数说明：
    ---------
    model_name : str
        模型名称，用于保存和加载时识别

    vocab_size : int
        词表大小，决定了模型能认识多少个不同的token
        中文常用字约6000个，加上标点和特殊符号，6400足够日常使用

    emb_dim : int
        嵌入维度（Embedding Dimension）
        每个token会被映射成一个 emb_dim 维的向量
        维度越大，表达能力越强，但计算量也越大

    num_heads : int
        多头注意力的头数
        将 emb_dim 分成 num_heads 份，每个头独立计算注意力
        要求 emb_dim 能被 num_heads 整除

    num_layers : int
        Transformer Block 的层数
        层数越多，模型越深，理解能力越强
        但也更难训练，容易过拟合

    context_size : int
        上下文窗口大小，模型一次能"看到"的最大token数
        越大能处理越长的文本，但显存占用呈平方增长

    dropout : float
        Dropout 比例，训练时随机丢弃神经元的概率
        用于防止过拟合，推理时应设为0

    学习相关参数：
    -------------
    learning_rate : float
        学习率，控制每次参数更新的步长
        太大会震荡，太小收敛慢

    batch_size : int
        批次大小，每次训练用多少条数据
        越大训练越稳定，但需要更多显存

    num_epochs : int
        训练轮数，整个数据集训练多少遍
    """

    # ==========================================
    # 模型基本信息
    # ==========================================
    model_name: str = "my_llm"
    version: str = "1.0.0"

    # ==========================================
    # 模型结构参数（这些参数决定了模型的"身材"）
    # ==========================================

    # 词表大小：模型的"词汇量"
    # 6400 = 常用中文字(~6000) + 标点符号 + 特殊token
    vocab_size: int = 6400

    # 嵌入维度：每个字被表示成多少维的向量
    # 256维是一个适合CPU训练的小规模配置
    # GPT-2 small 用的是 768维，GPT-3 用的是 12288维
    emb_dim: int = 256

    # 注意力头数：多头注意力中有多少个"专家"
    # 每个头关注文本的不同方面（如语法、语义、位置关系等）
    # 要求 emb_dim % num_heads == 0
    num_heads: int = 4

    # 每个头的维度（自动计算）
    # head_dim = emb_dim / num_heads = 256 / 4 = 64
    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.emb_dim // self.num_heads

    # Transformer层数：模型的"深度"
    # 层数越多，理解能力越强，但训练难度也越大
    # GPT-2 small 有12层，GPT-3 有96层
    num_layers: int = 4

    # 上下文长度：模型能"记住"多长的文本
    # 256 tokens 对于短对话足够
    # GPT-4 可以处理 128K tokens
    context_size: int = 256

    # Dropout比例：随机丢弃神经元的概率
    # 用于防止过拟合（背答案）
    # 推理时应设为0
    dropout: float = 0.1

    # ==========================================
    # 训练参数
    # ==========================================

    # 学习率：参数更新的"步子大小"
    # 太大容易震荡，太小收敛慢
    # 3e-4 是Adam优化器的经典起点
    learning_rate: float = 3e-4

    # 批次大小：每次用多少条数据更新参数
    # 越大训练越稳定，但需要更多内存
    batch_size: int = 32

    # 训练轮数：整个数据集过多少遍
    num_epochs: int = 10

    # ==========================================
    # 特殊Token ID
    # ==========================================

    # 填充token：用于对齐不同长度的序列
    pad_token_id: int = 0

    # 未知token：遇到词表外的字符时使用
    unk_token_id: int = 1

    # 序列开始token：标记文本的开头
    bos_token_id: int = 2

    # 序列结束token：标记文本的结尾
    eos_token_id: int = 3

    # SFT对话相关token
    im_start_token_id: int = 4  # <|im_start|>
    im_end_token_id: int = 5    # <|im_end|>

    def __post_init__(self):
        """初始化后的验证"""
        # 确保 emb_dim 能被 num_heads 整除
        assert self.emb_dim % self.num_heads == 0, \
            f"emb_dim ({self.emb_dim}) 必须能被 num_heads ({self.num_heads}) 整除"

    def save(self, path: str):
        """
        保存配置到JSON文件

        参数：
            path: 保存路径
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        print(f"配置已保存到: {path}")

    @classmethod
    def load(cls, path: str) -> 'MyLLMConfig':
        """
        从JSON文件加载配置

        参数：
            path: 配置文件路径

        返回：
            MyLLMConfig 实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 移除 property 字段（如果存在）
        data.pop('head_dim', None)
        return cls(**data)

    def __str__(self) -> str:
        """打印配置信息"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    MyLLM 配置                          ║
╠══════════════════════════════════════════════════════════════╣
║  模型名称: {self.model_name:<20} 版本: {self.version:<15} ║
╠══════════════════════════════════════════════════════════════╣
║  【模型结构】                                                 ║
║  · 词表大小 (vocab_size):     {self.vocab_size:<10} 个token          ║
║  · 嵌入维度 (emb_dim):        {self.emb_dim:<10} 维               ║
║  · 注意力头数 (num_heads):    {self.num_heads:<10} 个               ║
║  · 每头维度 (head_dim):       {self.head_dim:<10} 维               ║
║  · Transformer层数:           {self.num_layers:<10} 层               ║
║  · 上下文长度 (context_size): {self.context_size:<10} tokens          ║
║  · Dropout比例:               {self.dropout:<10.2f}                ║
╠══════════════════════════════════════════════════════════════╣
║  【训练参数】                                                 ║
║  · 学习率:     {self.learning_rate:<15}                        ║
║  · 批次大小:   {self.batch_size:<15}                              ║
║  · 训练轮数:   {self.num_epochs:<15}                              ║
╠══════════════════════════════════════════════════════════════╣
║  【预估参数量】                                               ║
║  · 约 {self._estimate_params() / 1e6:.2f}M 参数                              ║
╚══════════════════════════════════════════════════════════════╝
"""

    def _estimate_params(self) -> int:
        """
        估算模型参数量

        计算公式（简化版）：
        1. Token Embedding: vocab_size * emb_dim
        2. Position Embedding: context_size * emb_dim
        3. 每层 Transformer Block:
           - Attention (Q,K,V,O): 4 * emb_dim * emb_dim
           - FFN: 2 * emb_dim * (4 * emb_dim) = 8 * emb_dim^2
           - LayerNorm: 2 * 2 * emb_dim = 4 * emb_dim
        4. 输出层: emb_dim * vocab_size
        """
        # Embedding 层
        token_emb = self.vocab_size * self.emb_dim
        pos_emb = self.context_size * self.emb_dim

        # 每层 Transformer
        attn_params = 4 * self.emb_dim * self.emb_dim  # Q, K, V, O
        ffn_params = 8 * self.emb_dim * self.emb_dim   # 两个线性层
        norm_params = 4 * self.emb_dim                  # 两个 LayerNorm
        layer_params = attn_params + ffn_params + norm_params

        # 所有层
        all_layers = self.num_layers * layer_params

        # 输出层
        output = self.emb_dim * self.vocab_size

        # 最终 LayerNorm
        final_norm = 2 * self.emb_dim

        total = token_emb + pos_emb + all_layers + output + final_norm
        return total


# ==========================================
# RLHF 训练配置
# ==========================================

@dataclass
class RLHFTrainConfig:
    """
    RLHF 训练配置

    参数说明：
    --------
    reward_model_epochs : int
        奖励模型训练轮数

    reward_model_lr : float
        奖励模型学习率

    ppo_epochs : int
        PPO 更新轮数

    ppo_lr : float
        PPO 学习率

    clip_ratio : float
        PPO 裁剪系数

    kl_coef : float
        KL 散度惩罚系数

    num_episodes : int
        RLHF 训练的 episode 数

    batch_size : int
        每个 episode 的样本数

    max_new_tokens : int
        生成时的最大新 token 数
    """
    # 奖励模型训练
    reward_model_epochs: int = 3
    reward_model_lr: float = 1e-5
    reward_model_batch_size: int = 4

    # PPO 训练
    ppo_epochs: int = 4
    ppo_lr: float = 1e-5
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # 训练循环
    num_episodes: int = 100
    batch_size: int = 8
    max_new_tokens: int = 64

    # 其他
    max_grad_norm: float = 1.0
    gamma: float = 0.99
    lam: float = 0.95


@dataclass
class RLVFTrainConfig:
    """
    RLVF 训练配置

    参数说明：
    --------
    num_iterations : int
        训练迭代次数

    samples_per_task : int
        每个任务的采样次数

    correct_reward : float
        答案正确时的奖励

    incorrect_reward : float
        答案错误时的奖励

    learning_rate : float
        学习率

    max_new_tokens : int
        生成时的最大新 token 数
    """
    num_iterations: int = 50
    samples_per_task: int = 2
    batch_size: int = 4

    correct_reward: float = 1.0
    incorrect_reward: float = -0.5

    learning_rate: float = 1e-5
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    max_grad_norm: float = 1.0

    max_new_tokens: int = 32
    temperature: float = 0.7


# ==========================================
# 预设配置
# ==========================================

def get_mini_config() -> MyLLMConfig:
    """
    迷你版配置 (~5M参数)
    适合 CPU 训练，几分钟可见效果
    """
    return MyLLMConfig(
        model_name="my_llm-mini",
        vocab_size=6400,
        emb_dim=256,
        num_heads=4,
        num_layers=4,
        context_size=256,
        dropout=0.1,
    )


def get_small_config() -> MyLLMConfig:
    """
    小型版配置 (~50M参数)
    需要 GPU，效果更好
    """
    return MyLLMConfig(
        model_name="my_llm-small",
        vocab_size=6400,
        emb_dim=512,
        num_heads=8,
        num_layers=8,
        context_size=512,
        dropout=0.1,
    )


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 创建默认配置并打印
    config = get_mini_config()
    print(config)

    # 保存配置
    config.save("checkpoints/config.json")

    # 加载配置
    loaded_config = MyLLMConfig.load("checkpoints/config.json")
    print("配置加载成功！")
