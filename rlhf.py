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

        详细说明：
        ---------
        自回归语言模型的特性是：位置 t 的输出 logits 预测的是位置 t+1 的 token。

        举例说明（假设 response_start=5）：
        ┌─────────────────────────────────────────────────────────────┐
        │ 位置索引:    0     1     2     3     4  │  5     6     7    │
        │ Token:     [BOS] [你]  [好]  [吗]  [？] │ [我]  [很]  [好]   │
        │            ←───── prompt ─────────────→ │ ←── response ──→  │
        │                                         │                    │
        │ Logits:    L0    L1    L2    L3    L4   │ L5    L6    L7     │
        │ 预测目标:   ↓     ↓     ↓     ↓     ↓   │  ↓     ↓     ↓     │
        │           [你]  [好]  [吗]  [？]  [我]  │ [很]  [好]  [EOS]  │
        └─────────────────────────────────────────────────────────────┘

        为了计算 response 部分（位置 5,6,7）的生成概率：
        - L4 预测 token[5]="我" → 需要 logits[4]
        - L5 预测 token[6]="很" → 需要 logits[5]
        - L6 预测 token[7]="好" → 需要 logits[6]

        因此切片 logits[:, response_start-1:-1, :] 即 logits[:, 4:7, :]
        对应的目标是 full_ids[:, response_start:] 即 tokens[5:8]

        数学公式：
        ---------
        对于 token 序列 [t_1, t_2, ..., t_n]，其联合概率为：
        P(t_1, t_2, ..., t_n) = ∏_{i=1}^{n} P(t_i | t_1, ..., t_{i-1})

        对数概率：
        log P = Σ_{i=1}^{n} log P(t_i | t_1, ..., t_{i-1})
        """
        with torch.set_grad_enabled(model.training):
            # 前向传播：获取所有位置的 logits
            # logits 形状: [batch_size, seq_len, vocab_size]
            logits, _ = model(full_ids)

            # ============================================================
            # 关键步骤：提取预测 response 的 logits
            # ============================================================
            # logits[t] 预测 token[t+1]（自回归特性）
            # 所以要预测 response（从 response_start 开始），需要：
            #   - 起始：logits[response_start - 1]（预测第一个 response token）
            #   - 结束：logits[-2]（预测最后一个 response token，即 full_ids[-1]）
            # 切片 [response_start-1 : -1] 正好获取这些位置
            response_logits = logits[:, response_start - 1:-1, :]  # [1, response_len, vocab]

            # 获取实际生成的 response tokens 作为预测目标
            response_tokens = full_ids[:, response_start:]  # [1, response_len]

            # ============================================================
            # 计算每个 token 的对数概率
            # ============================================================
            # log_softmax 将 logits 转换为对数概率分布
            # log P(token | context) = log_softmax(logits)
            log_probs = F.log_softmax(response_logits, dim=-1)  # [1, response_len, vocab]

            # 使用 gather 提取实际生成 token 对应的对数概率
            # gather 操作示意：
            #   log_probs[batch, pos, :] 是一个 vocab_size 的向量
            #   response_tokens[batch, pos] 是实际 token 的索引
            #   gather 取出 log_probs[batch, pos, response_tokens[batch, pos]]
            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=response_tokens.unsqueeze(-1)  # [1, response_len, 1]
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
                # 计算 PPO 损失（核心算法）
                # ==========================================
                #
                # PPO 的核心思想：限制策略更新幅度，防止灾难性更新
                #
                # 数学推导：
                # ---------
                # 1. 策略梯度定理告诉我们：
                #    ∇J(θ) = E[∇log π_θ(a|s) * A(s,a)]
                #
                # 2. 重要性采样（使用旧策略的数据）：
                #    ∇J(θ) = E_{π_old}[r(θ) * A], 其中 r(θ) = π_new/π_old
                #
                # 3. PPO-Clip 的目标函数：
                #    L_CLIP = E[min(r*A, clip(r, 1-ε, 1+ε)*A)]
                #
                # 为什么要裁剪？
                # ------------
                # - 如果 A > 0（好动作），我们想增大 π_new，但不希望 r 太大
                # - 如果 A < 0（坏动作），我们想减小 π_new，但不希望 r 太小
                # - 裁剪保证 r 在 [1-ε, 1+ε] 范围内，限制单次更新幅度

                # ============================================================
                # Step 1: 计算概率比 r(θ) = π_new(a|s) / π_old(a|s)
                # ============================================================
                # 由于我们有对数概率，使用 exp(log_new - log_old) = new/old
                # 举例：如果 new_log_prob = -2, old_log_prob = -3
                #       ratio = exp(-2 - (-3)) = exp(1) ≈ 2.718
                #       这意味着新策略选择该动作的概率是旧策略的 2.718 倍
                ratio = torch.exp(new_log_probs - old_log_probs)

                # ============================================================
                # Step 2: 计算裁剪后的概率比
                # ============================================================
                # clip_ratio 通常设为 0.2，即 r 被限制在 [0.8, 1.2]
                # 这确保新策略不会偏离旧策略太远
                #
                # 可视化：
                #        0.8      1.0      1.2
                #    ─────┼────────┼────────┼─────
                #         │        │        │
                #    裁剪区 ←───────────────→ 裁剪区
                #              有效更新区
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.rlhf_config.clip_ratio,  # 下界 (如 0.8)
                    1 + self.rlhf_config.clip_ratio   # 上界 (如 1.2)
                )

                # ============================================================
                # Step 3: 计算策略损失（PPO-Clip 核心）
                # ============================================================
                # L = min(r * A, clip(r) * A)
                #
                # 这个 min 操作的巧妙之处：
                # ┌─────────────────────────────────────────────────────────┐
                # │ 情况1: A > 0 (好动作，应该鼓励)                          │
                # │   - 我们希望增大 π_new，即 r > 1                         │
                # │   - 但 min 操作限制了收益：当 r > 1+ε 时，取 clip(r)*A   │
                # │   - 超出范围后梯度为0，阻止继续增大                       │
                # │                                                          │
                # │ 情况2: A < 0 (坏动作，应该抑制)                          │
                # │   - 我们希望减小 π_new，即 r < 1                         │
                # │   - 但 min 操作限制了惩罚：当 r < 1-ε 时，取 clip(r)*A   │
                # │   - 超出范围后梯度为0，阻止继续减小                       │
                # └─────────────────────────────────────────────────────────┘
                #
                # 注意：这里加负号是因为我们做的是梯度下降（最小化损失）
                # 而 PPO 目标是最大化期望回报，所以需要取负
                policy_loss = -torch.min(
                    ratio * advantage,           # 未裁剪的目标
                    clipped_ratio * advantage    # 裁剪后的目标
                ).mean()

                # ============================================================
                # Step 4: 计算 KL 散度惩罚
                # ============================================================
                # KL 散度衡量新旧策略的差异程度
                # KL(π_old || π_new) ≈ E[log π_old - log π_new]
                #
                # 这是一个额外的正则化项，确保新策略不会偏离太远
                # kl_coef 控制惩罚强度（如 0.01）
                #
                # 注意：这里使用的是简化版 KL，真正的 KL 需要对所有动作求和
                kl_div = (old_log_probs - new_log_probs).mean()

                # ============================================================
                # Step 5: 计算总损失
                # ============================================================
                # 总损失 = 策略损失 + KL惩罚系数 * KL散度
                # 最小化这个损失会：
                #   1. 最大化好动作的概率（策略损失项）
                #   2. 惩罚偏离旧策略太远的更新（KL项）
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
