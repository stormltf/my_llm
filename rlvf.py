"""
MyLLM RLVF 训练模块
======================
这个文件实现了 RLVF（Reinforcement Learning from Verifiable Feedback）训练。

RLVF 与 RLHF 的区别：
--------------------
- RLHF：使用人类偏好训练奖励模型，再用奖励模型指导 RL
- RLVF：直接使用可验证的反馈（如数学答案正确性）作为奖励

RLVF 的优势：
-----------
1. 不需要人类标注偏好数据
2. 奖励信号客观、准确
3. 适用于有明确正确答案的任务

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

    工作流程：
    --------
    1. 采样：从任务列表中随机选择任务
    2. 生成：使用当前策略生成回答
    3. 验证：使用验证器判断答案正确性
    4. 更新：根据奖励进行策略梯度更新
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
        """
        total_reward = 0.0
        total_loss = 0.0
        total_correct = 0
        num_samples = 0

        self.policy.train()

        for task in tasks:
            for _ in range(self.rlvf_config.samples_per_task):
                # 生成并验证
                response, reward, full_ids, prompt_len = self.generate_and_verify(task)

                if full_ids.size(1) <= prompt_len:
                    continue

                # 确保策略模型处于训练模式（generate_and_verify 会将其设为 eval）
                self.policy.train()

                # 计算对数概率
                new_log_probs = self._compute_log_probs(
                    self.policy, full_ids, prompt_len
                )

                with torch.no_grad():
                    ref_log_probs = self._compute_log_probs(
                        self.ref_policy, full_ids, prompt_len
                    )

                # 策略梯度损失
                # L = -reward * log_prob
                advantage = reward  # 简化：直接使用奖励作为优势

                # 概率比
                ratio = torch.exp(new_log_probs - ref_log_probs)

                # PPO 裁剪
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.rlvf_config.clip_ratio,
                    1 + self.rlvf_config.clip_ratio
                )

                # 损失
                policy_loss = -torch.min(
                    ratio * advantage,
                    clipped_ratio * advantage
                ).mean()

                # KL 惩罚
                kl_div = (ref_log_probs - new_log_probs).mean()
                loss = policy_loss + self.rlvf_config.kl_coef * kl_div

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.rlvf_config.max_grad_norm
                )
                self.optimizer.step()

                # 统计
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
