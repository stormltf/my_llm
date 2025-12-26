"""
MyLLM 奖励模型
======================
这个文件实现了 RLHF 训练所需的奖励模型。

奖励模型的作用：
--------------
1. 评估 AI 回答的好坏
2. 从人类偏好数据中学习
3. 为 PPO 训练提供奖励信号

架构：
----
基于 MyLLM 的 Transformer 编码器，将输出层替换为标量奖励头。

训练目标：
--------
对于 (prompt, chosen, rejected) 三元组：
  reward(prompt + chosen) > reward(prompt + rejected)

使用 Bradley-Terry 模型损失函数训练。

作者：MyLLM Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import json
from copy import deepcopy

from config import MyLLMConfig
from model import MyLLM, LayerNorm, TransformerBlock


class RewardModel(nn.Module):
    """
    奖励模型

    基于 MyLLM 架构，将输出层替换为标量奖励头。

    工作流程：
    --------
    1. 输入 (prompt + response) 的 token 序列
    2. 通过 Transformer 编码
    3. 取最后一个 token 的表示
    4. 通过奖励头输出标量分数

    为什么取最后一个 token？
    --------------------
    因为在自回归模型中，最后一个 token 的表示
    已经"看过"了整个序列，包含了完整的上下文信息。
    """

    def __init__(self, config: MyLLMConfig):
        """
        参数：
            config: 模型配置
        """
        super().__init__()
        self.config = config

        # ==========================================
        # 复用 MyLLM 的 Transformer 编码器部分
        # ==========================================

        # Token Embedding
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.emb_dim
        )

        # Position Embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=config.context_size,
            embedding_dim=config.emb_dim
        )

        # Embedding Dropout
        self.embedding_dropout = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                emb_dim=config.emb_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                context_size=config.context_size
            )
            for _ in range(config.num_layers)
        ])

        # 最终 LayerNorm
        self.final_norm = LayerNorm(config.emb_dim)

        # ==========================================
        # 奖励头（替代原来的输出投影层）
        # ==========================================
        # 使用一个小型 MLP 将隐藏状态映射为标量奖励
        self.reward_head = nn.Sequential(
            nn.Linear(config.emb_dim, config.emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.emb_dim // 2, 1)
        )

        # 权重初始化
        self.apply(self._init_weights)

        print(f"奖励模型初始化完成！参数量: {self.num_parameters() / 1e6:.2f}M")

    def _init_weights(self, module: nn.Module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """返回模型参数总数"""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        参数：
            input_ids: 输入 Token ID，形状 [batch_size, seq_len]
            attention_mask: 注意力掩码（可选），形状 [batch_size, seq_len]
                          1 表示有效 token，0 表示 padding

        返回：
            rewards: 奖励分数，形状 [batch_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # ==========================================
        # Step 1: Embedding
        # ==========================================

        # Token Embedding
        token_emb = self.token_embedding(input_ids)

        # Position Embedding
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embedding(positions)

        # 相加
        x = token_emb + pos_emb
        x = self.embedding_dropout(x)

        # ==========================================
        # Step 2: Transformer Blocks
        # ==========================================

        for block in self.transformer_blocks:
            x = block(x)

        # ==========================================
        # Step 3: 最终归一化
        # ==========================================

        x = self.final_norm(x)

        # ==========================================
        # Step 4: 提取最后有效 token 的表示
        # ==========================================

        if attention_mask is not None:
            # 找到每个序列最后一个有效 token 的位置
            # attention_mask.sum(dim=-1) - 1 给出最后一个 1 的位置
            last_token_indices = attention_mask.sum(dim=-1).long() - 1
            last_token_indices = last_token_indices.clamp(min=0)

            # 使用 gather 提取对应位置的隐藏状态
            batch_indices = torch.arange(batch_size, device=device)
            last_hidden = x[batch_indices, last_token_indices]
        else:
            # 没有 mask，直接取最后一个位置
            last_hidden = x[:, -1, :]

        # ==========================================
        # Step 5: 计算奖励分数
        # ==========================================

        rewards = self.reward_head(last_hidden).squeeze(-1)

        return rewards

    @classmethod
    def from_pretrained(cls, base_model: MyLLM, config: MyLLMConfig) -> 'RewardModel':
        """
        从预训练的 MyLLM 模型初始化奖励模型

        参数：
            base_model: 预训练的 MyLLM 模型
            config: 模型配置

        返回：
            初始化的奖励模型
        """
        reward_model = cls(config)

        # 复制 Embedding 权重
        reward_model.token_embedding.load_state_dict(
            base_model.token_embedding.state_dict()
        )
        reward_model.position_embedding.load_state_dict(
            base_model.position_embedding.state_dict()
        )

        # 复制 Transformer Block 权重
        for i, block in enumerate(reward_model.transformer_blocks):
            block.load_state_dict(
                base_model.transformer_blocks[i].state_dict()
            )

        # 复制 final_norm 权重
        reward_model.final_norm.load_state_dict(
            base_model.final_norm.state_dict()
        )

        print("从预训练模型加载权重完成！")
        return reward_model


class RewardDataset(Dataset):
    """
    奖励模型训练数据集

    数据格式：
    --------
    [
        {
            "prompt": "问题文本",
            "chosen": "更好的回答",
            "rejected": "较差的回答"
        },
        ...
    ]

    处理流程：
    --------
    1. 将 prompt + response 拼接成完整序列
    2. 编码为 token ID
    3. 返回 chosen 和 rejected 两个版本
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 256
    ):
        """
        参数：
            data: 偏好数据列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 预处理数据
        self.processed_data = self._preprocess()

        print(f"奖励数据集创建完成！共 {len(self.processed_data)} 条数据")

    def _preprocess(self) -> List[Dict]:
        """预处理数据"""
        processed = []

        for item in self.data:
            prompt = item['prompt']
            chosen = item['chosen']
            rejected = item['rejected']

            # 构造完整的对话格式
            chosen_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{chosen}<|im_end|>"
            rejected_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{rejected}<|im_end|>"

            # 编码
            chosen_ids = self.tokenizer.encode(chosen_text)
            rejected_ids = self.tokenizer.encode(rejected_text)

            # 截断
            chosen_ids = chosen_ids[:self.max_length]
            rejected_ids = rejected_ids[:self.max_length]

            processed.append({
                'chosen_ids': chosen_ids,
                'rejected_ids': rejected_ids,
                'chosen_len': len(chosen_ids),
                'rejected_len': len(rejected_ids)
            })

        return processed

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.processed_data[idx]

        # 填充到相同长度
        chosen_ids = item['chosen_ids'] + [self.tokenizer.pad_token_id] * (
            self.max_length - len(item['chosen_ids'])
        )
        rejected_ids = item['rejected_ids'] + [self.tokenizer.pad_token_id] * (
            self.max_length - len(item['rejected_ids'])
        )

        # 创建 attention mask
        chosen_mask = [1] * item['chosen_len'] + [0] * (self.max_length - item['chosen_len'])
        rejected_mask = [1] * item['rejected_len'] + [0] * (self.max_length - item['rejected_len'])

        return {
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'chosen_mask': torch.tensor(chosen_mask, dtype=torch.long),
            'rejected_mask': torch.tensor(rejected_mask, dtype=torch.long)
        }


class RewardModelTrainer:
    """
    奖励模型训练器

    使用 Bradley-Terry 模型损失函数：
    --------------------------------
    L = -log(sigmoid(r_chosen - r_rejected))

    直觉理解：
    --------
    我们希望 r_chosen > r_rejected，
    即 r_chosen - r_rejected > 0，
    即 sigmoid(r_chosen - r_rejected) > 0.5，
    此时 -log(sigmoid(...)) 较小。

    这个损失函数来源于概率模型：
    P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    我们最大化这个概率，等价于最小化负对数似然。
    """

    def __init__(
        self,
        model: RewardModel,
        tokenizer,
        config: MyLLMConfig,
        learning_rate: float = 1e-5,
        num_epochs: int = 3
    ):
        """
        参数：
            model: 奖励模型
            tokenizer: 分词器
            config: 模型配置
            learning_rate: 学习率
            num_epochs: 训练轮数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"奖励模型训练器初始化完成！设备: {self.device}")

    def compute_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Bradley-Terry 损失

        参数：
            chosen_rewards: 好回答的奖励分数 [batch_size]
            rejected_rewards: 差回答的奖励分数 [batch_size]

        返回：
            损失值（标量）
        """
        # 计算奖励差
        reward_diff = chosen_rewards - rejected_rewards

        # Bradley-Terry 损失
        # L = -log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(reward_diff).mean()

        return loss

    def train(self, train_data: List[Dict], batch_size: int = 4) -> Dict:
        """
        训练奖励模型

        参数：
            train_data: 训练数据
            batch_size: 批次大小

        返回：
            训练历史记录
        """
        # 创建数据集和数据加载器
        dataset = RewardDataset(
            train_data,
            self.tokenizer,
            max_length=self.config.context_size
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

        history = {
            'loss': [],
            'accuracy': []
        }

        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch in dataloader:
                # 移动到设备
                chosen_ids = batch['chosen_ids'].to(self.device)
                rejected_ids = batch['rejected_ids'].to(self.device)
                chosen_mask = batch['chosen_mask'].to(self.device)
                rejected_mask = batch['rejected_mask'].to(self.device)

                # 前向传播
                chosen_rewards = self.model(chosen_ids, chosen_mask)
                rejected_rewards = self.model(rejected_ids, rejected_mask)

                # 计算损失
                loss = self.compute_loss(chosen_rewards, rejected_rewards)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 更新参数
                self.optimizer.step()

                # 统计
                total_loss += loss.item() * chosen_ids.size(0)
                correct = (chosen_rewards > rejected_rewards).sum().item()
                total_correct += correct
                total_samples += chosen_ids.size(0)

            # 计算平均指标
            avg_loss = total_loss / total_samples
            accuracy = total_correct / total_samples

            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            print(f"Epoch {epoch + 1}/{self.num_epochs} - "
                  f"Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

        return history

    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"奖励模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型"""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        print(f"奖励模型已从 {path} 加载")


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    from config import get_mini_config
    from tokenizer import MyLLMTokenizer

    print("=" * 60)
    print("奖励模型测试")
    print("=" * 60)

    # 1. 创建配置
    config = get_mini_config()

    # 2. 创建奖励模型
    reward_model = RewardModel(config)

    # 3. 测试前向传播
    print("\n测试前向传播:")
    batch_size = 2
    seq_len = 50
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    rewards = reward_model(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出奖励: {rewards}")
    print(f"输出形状: {rewards.shape}")

    # 4. 测试带 mask 的前向传播
    print("\n测试带 attention_mask 的前向传播:")
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 30:] = 0  # 第一个样本只有 30 个有效 token

    rewards = reward_model(input_ids, attention_mask)
    print(f"输出奖励: {rewards}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
