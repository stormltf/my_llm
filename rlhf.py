"""
MyLLM RLHF 训练模块
======================
这个文件实现了基于 PPO 算法的 RLHF 训练。

RLHF（Reinforcement Learning from Human Feedback）流程：
-------------------------------------------------------
1. 预训练语言模型（Pretrain）
2. 监督微调（SFT）
3. 训练奖励模型（Reward Model）
4. PPO 强化学习优化

PPO（Proximal Policy Optimization）算法：
-----------------------------------------
PPO 是一种策略梯度算法，通过限制策略更新幅度来保证训练稳定性。

核心公式：
L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

其中：
- r_t = π_new(a|s) / π_old(a|s)  (新旧策略的概率比)
- A_t = 优势函数（Advantage Function）
- ε = 裁剪系数（通常 0.2）

作者：MyLLM Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
import json

from config import MyLLMConfig
from model import MyLLM
from reward_model import RewardModel


@dataclass
class RLHFConfig:
    """
    RLHF 训练配置

    参数说明：
    --------
    clip_ratio : float
        PPO 裁剪系数，限制策略更新幅度
        通常设为 0.2

    kl_coef : float
        KL 散度惩罚系数
        防止新策略偏离原策略太远

    value_coef : float
        价值函数损失系数

    entropy_coef : float
        熵奖励系数，鼓励探索

    max_grad_norm : float
        梯度裁剪阈值

    ppo_epochs : int
        每批数据进行 PPO 更新的轮数

    mini_batch_size : int
        PPO 更新时的小批次大小

    learning_rate : float
        学习率

    gamma : float
        折扣因子

    lam : float
        GAE lambda 参数

    max_new_tokens : int
        生成时的最大新 token 数
    """
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    learning_rate: float = 1e-5
    gamma: float = 0.99
    lam: float = 0.95
    max_new_tokens: int = 64
    num_episodes: int = 100
    batch_size: int = 8


class PPOMemory:
    """
    PPO 经验回放缓冲区

    存储每个 episode 的轨迹数据：
    - prompts: 输入提示
    - responses: 生成的回复
    - old_log_probs: 旧策略的对数概率
    - rewards: 奖励
    - values: 价值估计
    - advantages: 优势函数值
    - returns: 回报
    """

    def __init__(self):
        self.clear()

    def clear(self):
        """清空缓冲区"""
        self.prompt_ids = []
        self.response_ids = []
        self.full_ids = []
        self.old_log_probs = []
        self.rewards = []
        self.advantages = []
        self.returns = []

    def add(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        full_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        reward: float
    ):
        """添加一条轨迹"""
        self.prompt_ids.append(prompt_ids)
        self.response_ids.append(response_ids)
        self.full_ids.append(full_ids)
        self.old_log_probs.append(old_log_probs)
        self.rewards.append(reward)

    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95):
        """
        计算优势函数（简化版本）

        对于单步奖励，优势 = 奖励 - 基线
        基线使用批次平均奖励
        """
        rewards = torch.tensor(self.rewards, dtype=torch.float32)

        # 使用简单的基线：批次平均奖励
        baseline = rewards.mean()

        # 优势 = 奖励 - 基线
        self.advantages = (rewards - baseline).tolist()

        # 回报 = 奖励（单步情况）
        self.returns = self.rewards.copy()

    def __len__(self):
        return len(self.rewards)


class PPOTrainer:
    """
    PPO 训练器

    负责 RLHF 的强化学习训练阶段。

    工作流程：
    --------
    1. 采样：使用当前策略生成回复
    2. 评估：使用奖励模型评分
    3. 计算优势：计算 GAE 优势函数
    4. 更新：多轮 PPO 更新
    """

    def __init__(
        self,
        policy_model: MyLLM,
        reward_model: RewardModel,
        tokenizer,
        config: MyLLMConfig,
        rlhf_config: RLHFConfig
    ):
        """
        参数：
            policy_model: 策略模型（待优化的 LLM）
            reward_model: 奖励模型
            tokenizer: 分词器
            config: 模型配置
            rlhf_config: RLHF 训练配置
        """
        self.policy = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.rlhf_config = rlhf_config

        # 创建参考模型（冻结的原策略副本）
        # 用于计算 KL 散度
        self.ref_policy = deepcopy(policy_model)
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=rlhf_config.learning_rate,
            weight_decay=0.01
        )

        # 经验缓冲区
        self.memory = PPOMemory()

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        self.ref_policy.to(self.device)
        self.reward_model.to(self.device)
        self.reward_model.eval()

        print(f"PPO 训练器初始化完成！设备: {self.device}")

    def _compute_log_probs(
        self,
        model: MyLLM,
        full_ids: torch.Tensor,
        response_start: int
    ) -> torch.Tensor:
        """
        计算生成的 token 序列的对数概率

        参数：
            model: 语言模型
            full_ids: 完整序列 [1, seq_len]
            response_start: 回复开始的位置

        返回：
            log_probs: 回复部分每个 token 的对数概率
        """
        with torch.set_grad_enabled(model.training):
            # 前向传播
            logits, _ = model(full_ids)

            # 获取回复部分的 logits（用于预测下一个 token）
            # logits[t] 预测 token[t+1]
            response_logits = logits[:, response_start - 1:-1, :]  # [1, response_len, vocab]

            # 获取实际生成的 token
            response_tokens = full_ids[:, response_start:]  # [1, response_len]

            # 计算对数概率
            log_probs = F.log_softmax(response_logits, dim=-1)

            # 提取实际 token 的对数概率
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=response_tokens.unsqueeze(-1)
            ).squeeze(-1)  # [1, response_len]

            return token_log_probs

    def generate_and_collect(self, prompts: List[str]) -> int:
        """
        生成回复并收集经验

        参数：
            prompts: 提示列表

        返回：
            收集的样本数
        """
        self.policy.eval()
        collected = 0

        for prompt in prompts:
            # 构造输入
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            prompt_ids = self.tokenizer.encode(formatted_prompt)
            prompt_tensor = torch.tensor([prompt_ids], device=self.device)
            prompt_len = len(prompt_ids)

            # 生成回复
            with torch.no_grad():
                generated = self.policy.generate(
                    prompt_tensor,
                    max_new_tokens=self.rlhf_config.max_new_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    eos_token_id=self.config.im_end_token_id
                )

            full_ids = generated  # [1, total_len]
            response_ids = full_ids[:, prompt_len:]  # [1, response_len]

            if response_ids.size(1) == 0:
                continue

            # 计算旧策略的对数概率
            with torch.no_grad():
                old_log_probs = self._compute_log_probs(
                    self.policy, full_ids, prompt_len
                )

            # 计算奖励
            with torch.no_grad():
                reward = self.reward_model(full_ids).item()

            # 添加到经验缓冲区
            self.memory.add(
                prompt_ids=prompt_tensor,
                response_ids=response_ids,
                full_ids=full_ids,
                old_log_probs=old_log_probs,
                reward=reward
            )
            collected += 1

        return collected

    def ppo_update(self) -> Dict:
        """
        执行 PPO 更新

        返回：
            训练指标
        """
        if len(self.memory) == 0:
            return {}

        self.policy.train()

        # 计算优势
        self.memory.compute_advantages(
            self.rlhf_config.gamma,
            self.rlhf_config.lam
        )

        total_loss = 0.0
        total_policy_loss = 0.0
        total_kl = 0.0
        num_updates = 0

        # 多轮 PPO 更新
        for _ in range(self.rlhf_config.ppo_epochs):
            for i in range(len(self.memory)):
                full_ids = self.memory.full_ids[i].to(self.device)
                old_log_probs = self.memory.old_log_probs[i].to(self.device)
                advantage = self.memory.advantages[i]
                prompt_len = self.memory.prompt_ids[i].size(1)

                # 计算新策略的对数概率
                new_log_probs = self._compute_log_probs(
                    self.policy, full_ids, prompt_len
                )

                # 计算参考策略的对数概率（用于 KL 散度）
                with torch.no_grad():
                    ref_log_probs = self._compute_log_probs(
                        self.ref_policy, full_ids, prompt_len
                    )

                # ==========================================
                # 计算 PPO 损失
                # ==========================================

                # 概率比
                ratio = torch.exp(new_log_probs - old_log_probs)

                # 裁剪目标
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.rlhf_config.clip_ratio,
                    1 + self.rlhf_config.clip_ratio
                )

                # 策略损失（取最小值）
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                ).mean()

                # KL 散度惩罚
                kl_div = (old_log_probs - new_log_probs).mean()

                # 总损失
                loss = policy_loss + self.rlhf_config.kl_coef * kl_div

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.rlhf_config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_kl += kl_div.item()
                num_updates += 1

        # 清空经验缓冲区
        self.memory.clear()

        return {
            'loss': total_loss / max(num_updates, 1),
            'policy_loss': total_policy_loss / max(num_updates, 1),
            'kl_div': total_kl / max(num_updates, 1)
        }

    def train(self, prompts: List[str]) -> Dict:
        """
        完整的 RLHF 训练循环

        参数：
            prompts: 用于训练的提示列表

        返回：
            训练历史
        """
        history = {
            'episode': [],
            'reward': [],
            'loss': [],
            'kl_div': []
        }

        num_episodes = self.rlhf_config.num_episodes
        batch_size = self.rlhf_config.batch_size

        for episode in range(num_episodes):
            # 随机采样一批提示
            batch_prompts = [
                prompts[i % len(prompts)]
                for i in range(episode * batch_size, (episode + 1) * batch_size)
            ]

            # 生成并收集经验
            collected = self.generate_and_collect(batch_prompts)

            if collected == 0:
                continue

            # 计算平均奖励
            avg_reward = sum(self.memory.rewards) / len(self.memory.rewards)

            # PPO 更新
            metrics = self.ppo_update()

            # 记录历史
            history['episode'].append(episode)
            history['reward'].append(avg_reward)
            history['loss'].append(metrics.get('loss', 0))
            history['kl_div'].append(metrics.get('kl_div', 0))

            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Reward: {avg_reward:.4f} - "
                      f"Loss: {metrics.get('loss', 0):.4f} - "
                      f"KL: {metrics.get('kl_div', 0):.4f}")

        return history

    def save_model(self, path: str):
        """保存策略模型"""
        torch.save(self.policy.state_dict(), path)
        print(f"RLHF 模型已保存到: {path}")

    def load_model(self, path: str):
        """加载策略模型"""
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        print(f"RLHF 模型已从 {path} 加载")


# ==========================================
# 便捷函数
# ==========================================

def load_prompts_from_sft(sft_data_path: str) -> List[str]:
    """
    从 SFT 数据中提取提示

    参数：
        sft_data_path: SFT 数据文件路径

    返回：
        提示列表
    """
    with open(sft_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompts = [item['user'] for item in data]
    return prompts


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    from config import get_mini_config

    print("=" * 60)
    print("RLHF 模块测试")
    print("=" * 60)

    # 1. 创建配置
    config = get_mini_config()
    rlhf_config = RLHFConfig(
        num_episodes=2,
        batch_size=2,
        ppo_epochs=1
    )

    # 2. 创建模型
    policy = MyLLM(config)
    reward_model = RewardModel(config)

    # 3. 创建简单的分词器模拟
    class SimpleTokenizer:
        def __init__(self):
            self.pad_token_id = 0

        def encode(self, text):
            return [ord(c) % 100 for c in text[:50]]

        def decode(self, ids):
            return ''.join([chr(i + 32) for i in ids])

    tokenizer = SimpleTokenizer()

    # 4. 创建 PPO 训练器
    trainer = PPOTrainer(
        policy_model=policy,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config,
        rlhf_config=rlhf_config
    )

    print("\nPPO 训练器创建成功！")
    print(f"策略模型参数量: {policy.num_parameters() / 1e6:.2f}M")
    print(f"奖励模型参数量: {reward_model.num_parameters() / 1e6:.2f}M")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
