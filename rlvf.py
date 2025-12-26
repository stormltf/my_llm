"""
MyLLM RLVF 训练模块
======================
这个文件实现了 RLVF（Reinforcement Learning from Verifiable Feedback）训练。

═══════════════════════════════════════════════════════════════════════════════
RLVF 与 RLHF 对比
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                           RLHF 流程                                          │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │人类偏好  │ → │ 训练奖励模型 │ → │  生成回答   │ → │  PPO 更新   │      │
│  │  数据   │    │   (RM)      │    │  获取奖励   │    │   策略     │      │
│  └─────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│       ↓                ↓                  ↓                  ↓              │
│   成本高           需要训练            RM 打分            更新模型          │
│   主观性           额外模型            (主观)                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           RLVF 流程                                          │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │可验证的 │ → │   无需RM    │ → │  生成回答   │ → │  PPO 更新   │      │
│  │ 任务   │    │   跳过!     │    │  直接验证   │    │   策略     │      │
│  └─────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│       ↓                                   ↓                  ↓              │
│   成本低                             验证器打分            更新模型          │
│   客观性                            (正确/错误)                              │
└─────────────────────────────────────────────────────────────────────────────┘

核心区别：
---------
┌────────────┬─────────────────────────────────────────────────────────────────┐
│   维度     │           RLHF              │             RLVF                 │
├────────────┼─────────────────────────────┼───────────────────────────────────┤
│  奖励来源  │  奖励模型 (主观)            │  验证器 (客观)                    │
│  数据需求  │  人类偏好标注               │  有正确答案的任务                 │
│  训练成本  │  需要额外训练 RM            │  无需额外训练                     │
│  适用场景  │  开放式对话、创意写作       │  数学、逻辑推理、代码             │
│  奖励精度  │  连续值 (0~1)               │  离散值 (0 或 1)                  │
└────────────┴─────────────────────────────┴───────────────────────────────────┘

RLVF 的优势：
-----------
1. 不需要人类标注偏好数据 → 降低成本
2. 奖励信号客观、准确 → 避免 Reward Hacking
3. 适用于有明确正确答案的任务 → 数学、编程、逻辑推理

支持的验证类型：
--------------
1. 数学验证：验证数学计算结果
2. 逻辑验证：验证逻辑推理答案

作者：MyLLM Team
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import json

from config import MyLLMConfig
from model import MyLLM


# ==========================================
# 验证器（Verifiers）
# ==========================================

class MathVerifier:
    """
    数学答案验证器

    功能：
    ----
    1. 从模型回答中提取数字
    2. 与期望答案比较
    3. 返回奖励（0或1）

    支持的格式：
    ----------
    - 纯数字：42
    - 带单位：42米
    - 表达式结果：等于42
    - 中文数字：四十二（暂不支持）
    """

    def __init__(self):
        # 数字提取正则表达式
        self.number_pattern = re.compile(r'-?\d+\.?\d*')

    def extract_numbers(self, text: str) -> List[float]:
        """从文本中提取所有数字"""
        matches = self.number_pattern.findall(text)
        numbers = []
        for m in matches:
            try:
                numbers.append(float(m))
            except ValueError:
                continue
        return numbers

    def verify(self, response: str, expected: str) -> float:
        """
        验证数学答案

        参数：
            response: 模型的回答
            expected: 期望的答案

        返回：
            奖励值（1.0 表示正确，0.0 表示错误）
        """
        # 提取期望答案中的数字
        expected_numbers = self.extract_numbers(expected)
        if not expected_numbers:
            return 0.0

        expected_value = expected_numbers[0]

        # 提取回答中的数字
        response_numbers = self.extract_numbers(response)
        if not response_numbers:
            return 0.0

        # 检查是否有匹配的数字
        for num in response_numbers:
            if abs(num - expected_value) < 1e-6:
                return 1.0

        return 0.0


class LogicVerifier:
    """
    逻辑推理验证器

    功能：
    ----
    1. 检查回答是否包含期望的关键词
    2. 检查回答中的数字是否匹配
    3. 返回奖励（0或1）

    验证策略：
    --------
    - 如果有期望数字，检查数字匹配
    - 如果有关键词列表，检查关键词匹配
    - 否则进行精确匹配
    """

    def __init__(self):
        self.number_pattern = re.compile(r'-?\d+\.?\d*')

    def extract_numbers(self, text: str) -> List[float]:
        """从文本中提取所有数字"""
        matches = self.number_pattern.findall(text)
        numbers = []
        for m in matches:
            try:
                numbers.append(float(m))
            except ValueError:
                continue
        return numbers

    def verify(
        self,
        response: str,
        expected: str,
        keywords: Optional[List[str]] = None
    ) -> float:
        """
        验证逻辑推理答案

        参数：
            response: 模型的回答
            expected: 期望的答案
            keywords: 可接受的关键词列表（可选）

        返回：
            奖励值（1.0 表示正确，0.0 表示错误）
        """
        response_lower = response.lower()
        expected_lower = expected.lower()

        # 1. 检查关键词
        if keywords:
            for keyword in keywords:
                if keyword.lower() in response_lower:
                    return 1.0

        # 2. 检查数字匹配
        expected_numbers = self.extract_numbers(expected)
        if expected_numbers:
            response_numbers = self.extract_numbers(response)
            for exp_num in expected_numbers:
                for resp_num in response_numbers:
                    if abs(exp_num - resp_num) < 1e-6:
                        return 1.0

        # 3. 检查期望答案是否在回答中
        if expected_lower in response_lower:
            return 1.0

        return 0.0


class CompositeVerifier:
    """
    组合验证器

    根据任务类型选择合适的验证器
    """

    def __init__(self):
        self.math_verifier = MathVerifier()
        self.logic_verifier = LogicVerifier()

    def verify(self, task: Dict, response: str) -> float:
        """
        验证回答

        参数：
            task: 任务字典，包含 type, expected_answer, keywords 等
            response: 模型的回答

        返回：
            奖励值
        """
        task_type = task.get('type', 'math')
        expected = task.get('expected_answer', '')
        keywords = task.get('keywords', None)

        if task_type == 'math':
            return self.math_verifier.verify(response, expected)
        elif task_type == 'logic':
            return self.logic_verifier.verify(response, expected, keywords)
        else:
            # 默认使用逻辑验证器
            return self.logic_verifier.verify(response, expected, keywords)


# ==========================================
# RLVF 配置
# ==========================================

@dataclass
class RLVFConfig:
    """
    RLVF 训练配置

    参数说明：
    --------
    clip_ratio : float
        PPO 裁剪系数

    kl_coef : float
        KL 散度惩罚系数

    correct_reward : float
        答案正确时的奖励

    incorrect_reward : float
        答案错误时的奖励（可以是负数作为惩罚）

    max_grad_norm : float
        梯度裁剪阈值

    learning_rate : float
        学习率

    max_new_tokens : int
        生成时的最大新 token 数

    num_iterations : int
        训练迭代次数

    samples_per_task : int
        每个任务的采样次数
    """
    clip_ratio: float = 0.2
    kl_coef: float = 0.01
    correct_reward: float = 1.0
    incorrect_reward: float = -0.5
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-5
    max_new_tokens: int = 32
    num_iterations: int = 50
    samples_per_task: int = 2
    temperature: float = 0.7


# ==========================================
# RLVF 训练器
# ==========================================

class RLVFTrainer:
    """
    RLVF 训练器

    使用可验证的反馈进行强化学习训练。

    ═══════════════════════════════════════════════════════════════════════════
    工作流程详解
    ═══════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RLVF 单步训练流程                                     │
    │                                                                          │
    │   ┌──────────────┐                                                       │
    │   │  1. 采样任务  │  从任务池随机选择一批任务                             │
    │   └──────┬───────┘                                                       │
    │          ↓                                                               │
    │   ┌──────────────┐                                                       │
    │   │  2. 生成回答  │  使用策略模型生成回答                                 │
    │   └──────┬───────┘                                                       │
    │          ↓                                                               │
    │   ┌──────────────┐     ┌─────────────────────────────────────┐          │
    │   │  3. 验证答案  │ ──→ │  MathVerifier / LogicVerifier      │          │
    │   └──────┬───────┘     │  返回 reward: 1.0 (正确) / -0.5 (错误)│          │
    │          ↓             └─────────────────────────────────────┘          │
    │   ┌──────────────┐                                                       │
    │   │  4. 计算损失  │  策略梯度 + KL 惩罚                                   │
    │   └──────┬───────┘                                                       │
    │          ↓                                                               │
    │   ┌──────────────┐                                                       │
    │   │  5. 更新策略  │  梯度下降，更新模型参数                               │
    │   └──────────────┘                                                       │
    └─────────────────────────────────────────────────────────────────────────┘

    验证器工作原理：
    ---------------
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        验证器架构                                        │
    │                                                                          │
    │                    ┌─────────────────┐                                   │
    │                    │ CompositeVerifier│                                   │
    │                    └────────┬────────┘                                   │
    │                             │                                            │
    │              ┌──────────────┴──────────────┐                             │
    │              ↓                             ↓                             │
    │     ┌─────────────────┐          ┌─────────────────┐                     │
    │     │  MathVerifier   │          │  LogicVerifier  │                     │
    │     │                 │          │                 │                     │
    │     │ • 提取数字      │          │ • 关键词匹配    │                     │
    │     │ • 数值比较      │          │ • 数字匹配      │                     │
    │     │ • 容差 1e-6    │          │ • 子串匹配      │                     │
    │     └─────────────────┘          └─────────────────┘                     │
    │                                                                          │
    │     示例：                                                               │
    │     ┌─────────────────────────────────────────────────────────────────┐ │
    │     │ 任务: "1+1等于几？"  期望答案: "2"                               │ │
    │     │ 模型回答: "1+1等于2"                                             │ │
    │     │ 验证: MathVerifier.extract_numbers("1+1等于2") → [1, 1, 2]      │ │
    │     │       期望值 = 2, 回答中有 2 → reward = 1.0 ✓                    │ │
    │     └─────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────────┘

    损失函数：
    ---------
    使用简化的 PPO 损失，与 RLHF 相同：

        L = L_policy + β * L_kl

    其中：
        L_policy = -min(r * A, clip(r, 1-ε, 1+ε) * A)
        L_kl = mean(log π_ref - log π_new)

    关键区别：在 RLVF 中，Advantage A 直接使用验证器的奖励值：
        A = +1.0  如果答案正确
        A = -0.5  如果答案错误（惩罚但不过于严厉）
    """

    def __init__(
        self,
        policy_model: MyLLM,
        tokenizer,
        config: MyLLMConfig,
        rlvf_config: RLVFConfig
    ):
        """
        参数：
            policy_model: 策略模型
            tokenizer: 分词器
            config: 模型配置
            rlvf_config: RLVF 训练配置
        """
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.config = config
        self.rlvf_config = rlvf_config

        # 创建参考模型（冻结）
        self.ref_policy = deepcopy(policy_model)
        for param in self.ref_policy.parameters():
            param.requires_grad = False

        # 验证器
        self.verifier = CompositeVerifier()

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=rlvf_config.learning_rate,
            weight_decay=0.01
        )

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        self.ref_policy.to(self.device)

        print(f"RLVF 训练器初始化完成！设备: {self.device}")

    def _compute_log_probs(
        self,
        model: MyLLM,
        full_ids: torch.Tensor,
        response_start: int
    ) -> torch.Tensor:
        """计算生成序列的对数概率"""
        with torch.set_grad_enabled(model.training):
            logits, _ = model(full_ids)

            response_logits = logits[:, response_start - 1:-1, :]
            response_tokens = full_ids[:, response_start:]

            log_probs = F.log_softmax(response_logits, dim=-1)

            token_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=response_tokens.unsqueeze(-1)
            ).squeeze(-1)

            return token_log_probs

    def _format_prompt(self, task: Dict) -> str:
        """格式化任务为提示"""
        prompt = task['prompt']
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def generate_and_verify(self, task: Dict) -> Tuple[str, float, torch.Tensor, int]:
        """
        生成回答并验证

        参数：
            task: 任务字典

        返回：
            (response_text, reward, full_ids, prompt_len)
        """
        # 格式化提示
        formatted_prompt = self._format_prompt(task)
        prompt_ids = self.tokenizer.encode(formatted_prompt)
        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
        prompt_len = len(prompt_ids)

        # 生成回答
        self.policy.eval()
        with torch.no_grad():
            generated = self.policy.generate(
                prompt_tensor,
                max_new_tokens=self.rlvf_config.max_new_tokens,
                temperature=self.rlvf_config.temperature,
                top_p=0.9,
                eos_token_id=self.config.im_end_token_id
            )

        full_ids = generated
        response_ids = full_ids[:, prompt_len:]

        # 解码回答
        response_text = self.tokenizer.decode(response_ids[0].tolist())

        # 验证答案
        reward = self.verifier.verify(task, response_text)

        # 调整奖励值
        if reward > 0.5:
            reward = self.rlvf_config.correct_reward
        else:
            reward = self.rlvf_config.incorrect_reward

        return response_text, reward, full_ids, prompt_len

    def train_step(self, tasks: List[Dict]) -> Dict:
        """
        单步训练

        参数：
            tasks: 任务列表

        返回：
            训练指标

        ═══════════════════════════════════════════════════════════════════════
        训练步骤详解
        ═══════════════════════════════════════════════════════════════════════

        对于每个任务，执行以下步骤：

        ┌─────────────────────────────────────────────────────────────────────┐
        │  Step 1: 生成回答                                                    │
        │  ─────────────────                                                   │
        │  prompt: "1+1等于几？"                                               │
        │  model.generate() → "1+1等于2"                                       │
        │                                                                      │
        │  Step 2: 验证答案                                                    │
        │  ─────────────────                                                   │
        │  Verifier("1+1等于2", expected="2") → reward = 1.0 ✓                │
        │                                                                      │
        │  Step 3: 计算对数概率                                                │
        │  ─────────────────────                                               │
        │  log π_new(response) = 每个 token 的对数概率之和                     │
        │  log π_ref(response) = 参考模型的对数概率（冻结）                    │
        │                                                                      │
        │  Step 4: 计算策略损失                                                │
        │  ─────────────────────                                               │
        │                                                                      │
        │     ┌──────────────────────────────────────────────────────────┐    │
        │     │  概率比: r = exp(log π_new - log π_ref)                   │    │
        │     │                                                           │    │
        │     │  裁剪: r_clip = clamp(r, 0.8, 1.2)                        │    │
        │     │                                                           │    │
        │     │  PPO 损失:                                                │    │
        │     │    L = -min(r × A, r_clip × A)                           │    │
        │     │                                                           │    │
        │     │  其中 A = reward（验证器的奖励）                          │    │
        │     │    A = +1.0 (正确) → 鼓励这个回答                         │    │
        │     │    A = -0.5 (错误) → 抑制这个回答                         │    │
        │     └──────────────────────────────────────────────────────────┘    │
        │                                                                      │
        │  Step 5: 添加 KL 惩罚                                               │
        │  ──────────────────────                                              │
        │  L_total = L_policy + β × KL(π_ref || π_new)                        │
        │                                                                      │
        │  防止策略偏离参考模型太远，保持生成的稳定性                          │
        │                                                                      │
        │  Step 6: 梯度更新                                                    │
        │  ─────────────────                                                   │
        │  optimizer.zero_grad()                                               │
        │  loss.backward()                                                     │
        │  clip_grad_norm_(max=1.0)  # 防止梯度爆炸                           │
        │  optimizer.step()                                                    │
        └─────────────────────────────────────────────────────────────────────┘

        奖励设计的思考：
        ---------------
        为什么 incorrect_reward = -0.5 而不是 -1.0？

        ┌─────────────────────────────────────────────────────────────────────┐
        │  如果惩罚太重（如 -1.0）：                                           │
        │    • 模型可能变得过于保守                                            │
        │    • 不敢尝试新的回答模式                                            │
        │    • 可能导致"模式崩塌"                                              │
        │                                                                      │
        │  使用 -0.5 的好处：                                                  │
        │    • 温和的惩罚，允许探索                                            │
        │    • 正确奖励(+1.0)与错误惩罚(-0.5)的比例为 2:1                       │
        │    • 给模型更多尝试的机会                                            │
        └─────────────────────────────────────────────────────────────────────┘
        """
        total_reward = 0.0
        total_loss = 0.0
        total_correct = 0
        num_samples = 0

        self.policy.train()

        for task in tasks:
            for _ in range(self.rlvf_config.samples_per_task):
                # ============================================================
                # Step 1 & 2: 生成回答并验证
                # ============================================================
                response, reward, full_ids, prompt_len = self.generate_and_verify(task)

                if full_ids.size(1) <= prompt_len:
                    continue

                # 确保策略模型处于训练模式（generate_and_verify 会将其设为 eval）
                self.policy.train()

                # ============================================================
                # Step 3: 计算对数概率
                # ============================================================
                # 新策略的对数概率（需要梯度）
                new_log_probs = self._compute_log_probs(
                    self.policy, full_ids, prompt_len
                )

                # 参考策略的对数概率（不需要梯度，作为基准）
                with torch.no_grad():
                    ref_log_probs = self._compute_log_probs(
                        self.ref_policy, full_ids, prompt_len
                    )

                # ============================================================
                # Step 4: 计算策略损失（PPO-Clip）
                # ============================================================
                # RLVF 简化：直接使用验证器奖励作为优势函数
                # 不需要像 RLHF 那样计算 GAE
                advantage = reward

                # 概率比: r = π_new / π_ref = exp(log π_new - log π_ref)
                ratio = torch.exp(new_log_probs - ref_log_probs)

                # PPO 裁剪：限制策略更新幅度
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.rlvf_config.clip_ratio,  # 下界 0.8
                    1 + self.rlvf_config.clip_ratio   # 上界 1.2
                )

                # PPO 目标：取裁剪前后的较小值
                # 这确保了无论优势正负，都能限制更新幅度
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                ).mean()

                # ============================================================
                # Step 5: 添加 KL 散度惩罚
                # ============================================================
                # KL(π_ref || π_new) ≈ E[log π_ref - log π_new]
                # 惩罚新策略偏离参考策略太远
                kl_div = (ref_log_probs - new_log_probs).mean()
                loss = policy_loss + self.rlvf_config.kl_coef * kl_div

                # ============================================================
                # Step 6: 梯度更新
                # ============================================================
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪：防止梯度爆炸导致训练不稳定
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.rlvf_config.max_grad_norm
                )
                self.optimizer.step()

                # ============================================================
                # 统计指标
                # ============================================================
                total_reward += reward
                total_loss += loss.item()
                if reward > 0:
                    total_correct += 1
                num_samples += 1

        return {
            'loss': total_loss / max(num_samples, 1),
            'avg_reward': total_reward / max(num_samples, 1),
            'accuracy': total_correct / max(num_samples, 1),
            'num_samples': num_samples
        }

    def train(self, tasks: List[Dict], batch_size: int = 4) -> Dict:
        """
        完整训练循环

        参数：
            tasks: 任务列表
            batch_size: 每次训练的任务数

        返回：
            训练历史
        """
        import random

        history = {
            'iteration': [],
            'loss': [],
            'reward': [],
            'accuracy': []
        }

        for iteration in range(self.rlvf_config.num_iterations):
            # 随机采样任务
            batch_tasks = random.sample(
                tasks,
                min(batch_size, len(tasks))
            )

            # 训练一步
            metrics = self.train_step(batch_tasks)

            # 记录历史
            history['iteration'].append(iteration)
            history['loss'].append(metrics['loss'])
            history['reward'].append(metrics['avg_reward'])
            history['accuracy'].append(metrics['accuracy'])

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.rlvf_config.num_iterations} - "
                      f"Loss: {metrics['loss']:.4f} - "
                      f"Reward: {metrics['avg_reward']:.4f} - "
                      f"Accuracy: {metrics['accuracy']:.4f}")

        return history

    def evaluate(self, tasks: List[Dict]) -> Dict:
        """
        评估模型

        参数：
            tasks: 任务列表

        返回：
            评估指标
        """
        self.policy.eval()

        total_correct = 0
        results = []

        with torch.no_grad():
            for task in tasks:
                response, reward, _, _ = self.generate_and_verify(task)

                is_correct = reward > 0
                if is_correct:
                    total_correct += 1

                results.append({
                    'prompt': task['prompt'],
                    'expected': task['expected_answer'],
                    'response': response,
                    'correct': is_correct
                })

        accuracy = total_correct / len(tasks) if tasks else 0

        return {
            'accuracy': accuracy,
            'total': len(tasks),
            'correct': total_correct,
            'results': results
        }

    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.policy.state_dict(), path)
        print(f"RLVF 模型已保存到: {path}")

    def load_model(self, path: str):
        """加载模型"""
        self.policy.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        print(f"RLVF 模型已从 {path} 加载")


# ==========================================
# 便捷函数
# ==========================================

def load_rlvf_data(data_path: str) -> List[Dict]:
    """
    加载 RLVF 训练数据

    参数：
        data_path: 数据文件路径

    返回：
        任务列表
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("RLVF 模块测试")
    print("=" * 60)

    # 1. 测试验证器
    print("\n【测试验证器】")

    math_verifier = MathVerifier()
    logic_verifier = LogicVerifier()

    # 数学验证测试
    print("\n数学验证测试:")
    test_cases = [
        ("答案是42", "42", True),
        ("结果等于 100", "100", True),
        ("我觉得是 50", "42", False),
        ("2+3=5，所以答案是5", "5", True),
    ]

    for response, expected, should_correct in test_cases:
        reward = math_verifier.verify(response, expected)
        status = "正确" if reward > 0.5 else "错误"
        match = "匹配" if (reward > 0.5) == should_correct else "不匹配"
        print(f"  回答: '{response}' | 期望: {expected} | 结果: {status} | {match}")

    # 逻辑验证测试
    print("\n逻辑验证测试:")
    logic_cases = [
        ("是的，猫会呼吸", "是", ["是", "会"], True),
        ("不，猫不会呼吸", "是", ["是", "会"], False),
        ("还剩9只羊", "9", None, True),
        ("还剩5只", "9", None, False),
    ]

    for response, expected, keywords, should_correct in logic_cases:
        reward = logic_verifier.verify(response, expected, keywords)
        status = "正确" if reward > 0.5 else "错误"
        match = "匹配" if (reward > 0.5) == should_correct else "不匹配"
        print(f"  回答: '{response}' | 期望: {expected} | 结果: {status} | {match}")

    # 2. 测试 RLVF 配置
    print("\n【测试 RLVF 配置】")
    rlvf_config = RLVFConfig(
        num_iterations=2,
        samples_per_task=1
    )
    print(f"裁剪系数: {rlvf_config.clip_ratio}")
    print(f"正确奖励: {rlvf_config.correct_reward}")
    print(f"错误奖励: {rlvf_config.incorrect_reward}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
