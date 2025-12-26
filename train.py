"""
大语言模型完整训练流程

实现完整的 5 阶段训练：
1. Pretrain (预训练) - 学习语言规律
2. SFT (监督微调) - 学习对话格式
3. Reward Model (奖励模型) - 学习人类偏好
4. RLHF (PPO) - 策略优化
5. RLVF - 可验证反馈强化学习

使用方法：
    # 完整训练
    python train.py

    # 跳过特定阶段
    python train.py --skip-pretrain --skip-sft

    # 只训练 RLHF/RLVF（需要已有 SFT 模型）
    python train.py --skip-pretrain --skip-sft --skip-reward
"""

# ============================================================
# 标准库导入
# ============================================================
import os                          # 操作系统接口：文件路径、目录操作
import argparse                    # 命令行参数解析
import json                        # JSON 文件读写
from datetime import datetime      # 日期时间处理
from typing import List, Dict, Optional  # 类型注解，提高代码可读性
from tqdm import tqdm              # 进度条显示库

# ============================================================
# PyTorch 相关导入
# ============================================================
import torch                       # PyTorch 核心库
import torch.nn as nn              # 神经网络模块（层、损失函数等）
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器

# ============================================================
# 本项目模块导入
# ============================================================
from model import GPT, GPTConfig, MyLLM       # GPT 模型定义
from config import MyLLMConfig, get_mini_config  # 配置类
from tokenizer import BPETokenizer            # BPE 分词器


# ==========================================
# 数据集类
# ==========================================

class PretrainDataset(Dataset):
    """
    预训练数据集

    功能：将原始文本转换为模型可用的训练样本

    预训练任务：语言建模（Language Modeling）
    --------------------------------------
    给定前 n 个词，预测第 n+1 个词

    数据处理流程：
    -------------
    原始文本: "我 喜欢 学习 人工智能"
        ↓ tokenize
    Token IDs: [101, 234, 567, 890, 123]
        ↓ 滑动窗口切分
    样本1: input=[101,234,567], target=[234,567,890]
    样本2: input=[234,567,890], target=[567,890,123]
    """

    def __init__(self, texts: List[str], tokenizer: BPETokenizer, seq_len: int):
        """
        初始化预训练数据集

        参数:
            texts: 原始文本列表，每个元素是一段文本
            tokenizer: BPE 分词器，用于将文本转为 token ID
            seq_len: 序列长度，每个训练样本的 token 数量
        """
        self.tokenizer = tokenizer  # 保存分词器引用
        self.seq_len = seq_len      # 保存序列长度
        self.samples = []           # 存储处理后的训练样本

        # ============================================================
        # Step 1: 将所有文本编码为 token ID 序列
        # ============================================================
        print("正在处理预训练数据...")
        all_token_ids = []          # 存储所有文本的 token ID（拼接成一个长序列）

        for text in tqdm(texts, desc="编码文本"):  # tqdm 显示进度条
            token_ids = tokenizer.encode(text)     # 将文本转为 token ID 列表
            all_token_ids.extend(token_ids)        # 追加到总序列中

        print(f"总共编码了 {len(all_token_ids)} 个 token")

        # ============================================================
        # Step 2: 使用滑动窗口切分训练样本
        # ============================================================
        # 自回归训练：用 token[i:i+seq_len] 预测 token[i+1:i+seq_len+1]
        #
        # 举例（seq_len=3）：
        #   all_token_ids = [A, B, C, D, E, F, G]
        #
        #   i=0: input=[A,B,C], target=[B,C,D]  # 用ABC预测BCD
        #   i=1: input=[B,C,D], target=[C,D,E]  # 用BCD预测CDE
        #   i=2: input=[C,D,E], target=[D,E,F]  # 用CDE预测DEF
        #   ...
        for i in range(0, len(all_token_ids) - seq_len - 1):
            # 输入序列：从位置 i 开始，取 seq_len 个 token
            input_ids = all_token_ids[i:i + seq_len]
            # 目标序列：从位置 i+1 开始，取 seq_len 个 token（向后偏移1位）
            target_ids = all_token_ids[i + 1:i + seq_len + 1]

            # 保存为字典格式
            self.samples.append({
                'input_ids': input_ids,
                'target_ids': target_ids
            })

        print(f"生成了 {len(self.samples)} 个训练样本")

    def __len__(self):
        """
        返回数据集大小

        PyTorch DataLoader 需要这个方法来知道有多少样本
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的样本

        参数:
            idx: 样本索引，0 到 len(dataset)-1

        返回:
            (input_tensor, target_tensor) 元组
            - input_tensor: 形状 [seq_len]，输入 token ID
            - target_tensor: 形状 [seq_len]，目标 token ID

        PyTorch DataLoader 会调用这个方法来获取每个样本
        """
        sample = self.samples[idx]
        return (
            # torch.tensor() 将 Python 列表转为 PyTorch 张量
            # dtype=torch.long 表示 64 位整数（token ID 必须是整数）
            torch.tensor(sample['input_ids'], dtype=torch.long),
            torch.tensor(sample['target_ids'], dtype=torch.long)
        )


class SFTDataset(Dataset):
    """
    SFT (Supervised Fine-Tuning) 数据集

    核心设计：只对 assistant 回复部分计算 loss

    为什么这样设计？
    ---------------
    1. 我们希望模型学会"如何回答"，而不是"如何提问"
    2. 用户的输入是已知的，不需要模型去预测
    3. 只在 assistant 部分计算 loss 可以：
       - 更高效地利用梯度更新
       - 避免模型学习复述用户输入
       - 让模型专注于生成高质量回复

    数据处理流程示意：
    -----------------
    原始对话:
        user: "你好"
        assistant: "你好！有什么可以帮助你的吗？"

    编码后的 token 序列:
        [<im_start>, user, \\n, 你, 好, <im_end>, \\n, <im_start>, assistant, \\n, 你, 好, ！, 有, ...]
        ├─────────────── user_part ─────────────────┤├────── assistant_part ──────┤
        │                                            │
        │        这部分 loss 设为 -1 (忽略)          │  这部分正常计算 loss
    """

    def __init__(self, data: List[Dict], tokenizer: BPETokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print("正在处理 SFT 数据...")
        for item in tqdm(data, desc="处理对话"):
            # ============================================================
            # Step 1: 分别编码用户和助手部分
            # ============================================================
            # 使用 ChatML 格式：<|im_start|>role\ncontent<|im_end|>
            # 这种格式让模型能够区分不同角色的发言
            user_part = f"<|im_start|>user\n{item['user']}<|im_end|>\n<|im_start|>assistant\n"
            assistant_part = f"{item['assistant']}<|im_end|>"

            user_ids = tokenizer.encode(user_part)
            assistant_ids = tokenizer.encode(assistant_part)

            # 完整序列 = user_part + assistant_part
            token_ids = user_ids + assistant_ids

            # ============================================================
            # Step 2: 截断处理（防止超过最大长度）
            # ============================================================
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                # 重新计算 user 部分长度（用于后续创建 mask）
                user_len = min(len(user_ids), max_length - 1)
            else:
                user_len = len(user_ids)

            # ============================================================
            # Step 3: 构造自回归训练样本
            # ============================================================
            # 自回归：用 token[0:n-1] 预测 token[1:n]
            #
            # 举例（假设 token_ids = [A, B, C, D, E]）：
            #   input_ids  = [A, B, C, D]     (前 n-1 个)
            #   target_ids = [B, C, D, E]     (后 n-1 个，向后偏移1位)
            #
            # 这样模型学习：给定 A 预测 B，给定 AB 预测 C，以此类推
            if len(token_ids) > 1:
                input_ids = token_ids[:-1]   # 去掉最后一个
                target_ids = token_ids[1:]   # 去掉第一个

                # ============================================================
                # Step 4: 创建 Loss Mask
                # ============================================================
                # 关键：只对 assistant 部分计算 loss
                #
                # 假设 user_len = 5，token 序列如下：
                #   位置:    0    1    2    3    4  │  5    6    7    8
                #   Token: [u1] [u2] [u3] [u4] [u5] │ [a1] [a2] [a3] [a4]
                #          ←───── user_part ─────→  │ ←── assistant_part ──→
                #
                # 自回归偏移后：
                #   input:   [u1] [u2] [u3] [u4] [u5] [a1] [a2] [a3]
                #   target:  [u2] [u3] [u4] [u5] [a1] [a2] [a3] [a4]
                #            ├─── 忽略(mask=-1) ──┤ ├─ 计算loss ─┤
                #            位置 0 到 user_len-2     位置 user_len-1 开始
                #
                # PyTorch 的 CrossEntropyLoss 会自动忽略 target=-100 的位置
                loss_mask = [-100] * (user_len - 1) + target_ids[user_len - 1:]

                self.samples.append({
                    'input_ids': input_ids,
                    'target_ids': loss_mask  # 使用带 mask 的 target
                })

        print(f"SFT 数据集大小: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['input_ids'], dtype=torch.long),
            torch.tensor(sample['target_ids'], dtype=torch.long)
        )


def collate_fn(batch):
    """
    自定义批次整理函数（Collate Function）

    为什么需要这个函数？
    ------------------
    1. DataLoader 默认会把多个样本堆叠成一个批次
    2. 但堆叠要求所有样本长度相同
    3. SFT 数据集中每个对话长度不同，需要手动处理

    处理流程：
    ---------
    输入 batch（假设 batch_size=3）：
        样本1: [A, B, C]        (长度 3)
        样本2: [D, E, F, G, H]  (长度 5)
        样本3: [I, J]           (长度 2)

    处理后（填充到最大长度 5）：
        样本1: [A, B, C, 0, 0]     target: [B, C, D, -1, -1]
        样本2: [D, E, F, G, H]     target: [E, F, G, H, I]
        样本3: [I, J, 0, 0, 0]     target: [J, K, -1, -1, -1]

    注意：target 中的 -1 会被 CrossEntropyLoss 忽略（ignore_index=-1）

    参数:
        batch: 列表，每个元素是 (input_ids, target_ids) 元组

    返回:
        (padded_inputs, padded_targets) 元组
        - padded_inputs: [batch_size, max_len]
        - padded_targets: [batch_size, max_len]
    """
    # 从 batch 中分离出所有的 input 和 target
    input_ids = [item[0] for item in batch]   # 列表，每个元素是一个张量
    target_ids = [item[1] for item in batch]

    # 找到这个批次中最长的序列长度
    max_len = max(len(ids) for ids in input_ids)

    # 准备填充后的列表
    padded_inputs = []
    padded_targets = []

    # 逐个样本进行填充
    for inp, tgt in zip(input_ids, target_ids):
        # 计算需要填充的长度
        pad_len = max_len - len(inp)

        # 填充 input：用 0（通常是 <PAD> token）
        # torch.cat 拼接两个张量
        padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))

        # 填充 target：用 -100（让 loss 函数忽略这些位置）
        # torch.full 创建一个填满指定值的张量
        padded_targets.append(torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)]))

    # torch.stack 将列表中的张量堆叠成一个批次张量
    # [tensor1, tensor2, tensor3] → [3, max_len]
    return torch.stack(padded_inputs), torch.stack(padded_targets)


# ==========================================
# 数据加载函数
# ==========================================
# 每个训练阶段需要不同格式的数据：
#
# 阶段         | 数据格式                    | 文件
# ------------|---------------------------|------------------
# Pretrain    | 纯文本                      | pretrain_data.txt
# SFT         | {"user": ..., "assistant": ...}  | sft_data.json
# Reward      | {"prompt": ..., "chosen": ..., "rejected": ...} | reward_data.json
# RLVF        | {"question": ..., "answer": ...} | rlvf_data.json

def load_pretrain_data() -> List[str]:
    """
    加载预训练数据

    数据格式：每行一段文本
    示例文件内容：
        我是一个人工智能助手
        人工智能是计算机科学的一个分支
        深度学习是机器学习的一种方法

    返回:
        文本列表，每个元素是一行文本
    """
    data_path = "data/pretrain_data.txt"  # 数据文件路径

    # 检查文件是否存在
    if os.path.exists(data_path):
        # 读取文件，按行分割，去除空行和首尾空白
        with open(data_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # 文件不存在时使用内置示例数据（用于测试）
        print("未找到预训练数据文件，使用内置示例数据")
        corpus = [
            "我 是 一个 人工智能 助手",
            "人工智能 是 计算机 科学 的 一个 分支",
            "深度 学习 是 机器 学习 的 一种 方法",
            "自然 语言 处理 让 计算机 理解 人类 语言",
            "大 语言 模型 可以 生成 流畅 的 文本",
        ] * 100  # 重复 100 次增加数据量
        return corpus


def load_sft_data() -> List[Dict]:
    """
    加载 SFT（监督微调）数据

    数据格式：JSON 数组，每个元素包含 user 和 assistant 字段
    示例文件内容：
        [
            {"user": "你好", "assistant": "你好！有什么可以帮助你的吗？"},
            {"user": "1+1等于多少", "assistant": "1+1等于2"}
        ]

    返回:
        对话列表，每个元素是 {"user": ..., "assistant": ...} 字典
    """
    data_path = "data/sft_data.json"

    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)  # json.load 直接解析文件为 Python 对象
    else:
        print("未找到 SFT 数据文件")
        return []  # 返回空列表表示无数据


def load_reward_data() -> List[Dict]:
    """
    加载奖励模型训练数据

    数据格式：每条数据包含一个 prompt 和两个回答（好的和差的）
    示例文件内容：
        [
            {
                "prompt": "如何学习编程？",
                "chosen": "建议从 Python 开始，它语法简洁...",
                "rejected": "编程很难学"
            }
        ]

    这种"偏好对比"数据用于训练奖励模型区分回答好坏

    返回:
        偏好数据列表
    """
    data_path = "data/reward_data.json"

    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("未找到奖励数据文件")
        return []


def load_rlvf_data() -> List[Dict]:
    """
    加载 RLVF（可验证反馈强化学习）数据

    数据格式：数学或逻辑问题，带有可验证的正确答案
    示例文件内容：
        [
            {"question": "2 + 3 = ?", "answer": "5"},
            {"question": "如果 x = 2，那么 x * 3 = ?", "answer": "6"}
        ]

    RLVF 的特点是答案可以自动验证（而不需要人工评估）

    返回:
        问答数据列表
    """
    data_path = "data/rlvf_data.json"

    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("未找到 RLVF 数据文件")
        return []


def train_pretrain(
    model: GPT,
    tokenizer: BPETokenizer,
    config: argparse.Namespace,
    device: torch.device
) -> Dict:
    """
    阶段 1：预训练

    目标：学习语言规律，预测下一个词
    """
    print("\n" + "=" * 60)
    print("阶段 1：预训练 (Pretrain)")
    print("=" * 60)

    # 加载数据
    corpus = load_pretrain_data()
    if not corpus:
        print("没有预训练数据，跳过")
        return {}

    dataset = PretrainDataset(corpus, tokenizer, seq_len=config.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    # ============================================================
    # 优化器 (Optimizer)
    # ============================================================
    # AdamW = Adam + Weight Decay（权重衰减）
    #
    # 为什么用 AdamW 而不是 SGD？
    # - Adam 自适应调整每个参数的学习率，收敛更快
    # - Weight Decay 是正则化手段，防止过拟合
    #
    # 参数说明：
    # - lr: 学习率，控制每次更新的步长
    # - weight_decay: 权重衰减系数，相当于 L2 正则化
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain_lr,
        weight_decay=0.01
    )

    # ============================================================
    # 学习率调度器 (Learning Rate Scheduler)
    # ============================================================
    # CosineAnnealingLR: 余弦退火学习率
    #
    # 学习率变化曲线：
    #   lr
    #    │
    #  max├───╮
    #    │    ╲
    #    │     ╲___
    #  min├────────╲____
    #    │              ╲
    #    └───────────────╲──→ epoch
    #    0               T_max
    #
    # 为什么要衰减学习率？
    # - 训练初期：大学习率快速找到好的方向
    # - 训练后期：小学习率精细调整，避免震荡
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.pretrain_epochs
    )

    history = {'loss': []}

    # model.train() 开启训练模式
    # 这会启用 Dropout 和 BatchNorm 的训练行为
    model.train()

    # ============================================================
    # 训练主循环
    # ============================================================
    #
    # 深度学习训练的整体流程：
    # ┌────────────────────────────────────────────────────────────┐
    # │                    Epoch 循环（遍历整个数据集）              │
    # │  ┌──────────────────────────────────────────────────────┐  │
    # │  │               Batch 循环（处理一个批次）               │  │
    # │  │                                                      │  │
    # │  │  1. 前向传播 ──→ 计算预测值和损失                      │  │
    # │  │  2. 反向传播 ──→ 计算梯度                             │  │
    # │  │  3. 梯度裁剪 ──→ 防止梯度爆炸                         │  │
    # │  │  4. 参数更新 ──→ 根据梯度调整权重                      │  │
    # │  │                                                      │  │
    # │  └──────────────────────────────────────────────────────┘  │
    # │  5. 更新学习率调度器                                        │
    # └────────────────────────────────────────────────────────────┘

    for epoch in range(config.pretrain_epochs):
        total_loss = 0
        # tqdm 是进度条库，显示训练进度
        progress_bar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch + 1}/{config.pretrain_epochs}")

        # ============================================================
        # Batch 循环：遍历数据集中的每个批次
        # ============================================================
        # dataloader 每次返回一个 batch 的数据：
        #   input_ids:  [batch_size, seq_len]，如 [16, 64]
        #   target_ids: [batch_size, seq_len]，如 [16, 64]
        for input_ids, target_ids in progress_bar:
            # --------------------------------------------------------
            # Step 0: 将数据移动到 GPU（如果有的话）
            # --------------------------------------------------------
            # .to(device) 将张量从 CPU 复制到 GPU
            # 神经网络计算在 GPU 上比 CPU 快 10-100 倍
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # --------------------------------------------------------
            # Step 1: 前向传播 (Forward Pass)
            # --------------------------------------------------------
            # 输入数据通过模型，得到预测结果和损失
            #
            # 数据流：
            #   input_ids [16, 64]
            #       ↓
            #   Embedding Layer（词向量）
            #       ↓
            #   Transformer Blocks × N（特征提取）
            #       ↓
            #   Output Layer（预测下一个词）
            #       ↓
            #   logits [16, 64, vocab_size]
            #       ↓
            #   CrossEntropyLoss（与 target_ids 比较）
            #       ↓
            #   loss（标量）
            _, loss = model(input_ids, target_ids)

            # --------------------------------------------------------
            # Step 2: 清零梯度
            # --------------------------------------------------------
            # 为什么要清零？
            # PyTorch 默认会累加梯度，如果不清零，梯度会越来越大
            # 这是为了支持"梯度累积"技术，但通常我们每个 batch 要清零
            optimizer.zero_grad()

            # --------------------------------------------------------
            # Step 3: 反向传播 (Backward Pass)
            # --------------------------------------------------------
            # 计算损失对每个参数的梯度
            #
            # 链式法则：
            #   ∂L/∂W = ∂L/∂output × ∂output/∂hidden × ∂hidden/∂W
            #
            # loss.backward() 自动计算所有参数的梯度
            # 梯度存储在 param.grad 中
            loss.backward()

            # --------------------------------------------------------
            # Step 4: 梯度裁剪 (Gradient Clipping)
            # --------------------------------------------------------
            # 防止"梯度爆炸"问题
            #
            # 什么是梯度爆炸？
            # - 在深层网络中，梯度通过链式法则相乘
            # - 如果每层梯度 > 1，累乘后会变得非常大
            # - 导致参数更新过大，模型发散
            #
            # 梯度裁剪原理：
            #   如果 ||gradients|| > max_norm:
            #       gradients = gradients × (max_norm / ||gradients||)
            #
            # 举例（max_norm=1.0）：
            #   原始梯度向量: [3, 4]，范数 = 5
            #   裁剪后: [3, 4] × (1/5) = [0.6, 0.8]，范数 = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # --------------------------------------------------------
            # Step 5: 参数更新
            # --------------------------------------------------------
            # 根据梯度调整模型参数
            #
            # Adam 更新公式（简化版）：
            #   m = β1 × m + (1-β1) × grad      # 一阶动量（梯度的移动平均）
            #   v = β2 × v + (1-β2) × grad²     # 二阶动量（梯度平方的移动平均）
            #   param = param - lr × m / (√v + ε)
            #
            # 每个参数都会被更新：
            #   W_new = W_old - lr × ∂L/∂W
            optimizer.step()

            # --------------------------------------------------------
            # 统计和显示
            # --------------------------------------------------------
            # .item() 将单元素张量转为 Python 数值
            total_loss += loss.item()
            # 在进度条右侧显示当前 batch 的 loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # ============================================================
        # Epoch 结束后的操作
        # ============================================================

        # 更新学习率（按余弦曲线衰减）
        scheduler.step()

        # 计算并记录平均损失
        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

    # 保存模型
    save_path = os.path.join(config.checkpoint_dir, "pretrain_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"预训练模型已保存: {save_path}")

    return history


def train_sft(
    model: GPT,
    tokenizer: BPETokenizer,
    config: argparse.Namespace,
    device: torch.device
) -> Dict:
    """
    阶段 2：监督微调 (SFT)

    目标：学习对话格式，获得指令遵循能力
    包含早停机制防止过拟合
    """
    print("\n" + "=" * 60)
    print("阶段 2：监督微调 (SFT)")
    print("=" * 60)

    # 加载数据
    sft_data = load_sft_data()
    if not sft_data:
        print("没有 SFT 数据，跳过")
        return {}

    dataset = SFTDataset(sft_data, tokenizer, max_length=config.context_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 优化器（使用较小的学习率）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.sft_lr,
        weight_decay=0.01
    )

    history = {'loss': []}
    model.train()

    # ============================================================
    # 早停机制 (Early Stopping) 配置
    # ============================================================
    #
    # 什么是早停？
    # -----------
    # 早停是防止过拟合的技术：当模型在验证集上的性能不再提升时停止训练。
    # 在 SFT 阶段特别重要，因为：
    #   1. SFT 数据集通常较小（几百到几千条）
    #   2. 模型很容易在小数据集上过拟合
    #   3. 过拟合后模型会"死记硬背"，泛化能力下降
    #
    # 我们的早停策略：
    # --------------
    #   1. 只有当 loss < min_loss_threshold 时才开始监控
    #      （避免在训练初期误判）
    #   2. 要求 loss 有"明显改善"（下降超过 0.01）
    #   3. 连续 patience 个 epoch 没有明显改善就停止
    #
    # 可视化（假设 min_loss_threshold=0.1, patience=5）：
    #
    #   Loss
    #    │
    #  1.0├────╮
    #    │    ╰──╮
    #  0.5├       ╰───╮
    #    │           ╰──╮
    #  0.1├─ ─ ─ ─ ─ ─ ─╰──╮────────────  ← 开始监控
    #    │                 ╰─╮ ╭─╮ ╭─╮
    #  0.05├                  ╰─╯ ╰─╯ ╰─→  ← 5次无改善，停止
    #    │
    #    └───┬───┬───┬───┬───┬───┬───┬───→ Epoch
    #        1   2   3   4   5   6   7   8

    best_loss = float('inf')         # 记录最佳 loss
    patience = 5                      # 容忍次数：连续几次无改善后停止
    patience_counter = 0              # 当前已连续无改善的次数
    min_loss_threshold = 0.1          # 开始监控的阈值（loss 低于此值才开始）
    improvement_threshold = 0.01      # 改善阈值：loss 需要下降超过此值才算"改善"
    #
    # 为什么 improvement_threshold = 0.01？
    # -----------------------------------
    # 1. 太小（如 0.001）：对噪声过于敏感，可能过早停止
    # 2. 太大（如 0.1）：可能错过最佳点，训练过久
    # 3. 0.01 是经验值：在大多数 SFT 任务中效果良好
    #    - 对于 loss 在 0.01-0.1 范围内，0.01 的变化约是 10%-100%
    #    - 这个幅度足够区分真正的改善和随机波动

    for epoch in range(config.sft_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"SFT Epoch {epoch + 1}/{config.sft_epochs}")

        for input_ids, target_ids in progress_bar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪：防止梯度爆炸，1.0 是常用的阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

        # ============================================================
        # 早停检查逻辑
        # ============================================================
        # 条件1：只有 loss 低于阈值才开始监控
        # 这是因为训练初期 loss 波动较大，不适合做早停判断
        if avg_loss < min_loss_threshold:

            # 条件2：检查是否有"明显改善"
            # best_loss - 0.01 表示 loss 需要比历史最佳低至少 0.01
            if avg_loss < best_loss - improvement_threshold:
                # 有明显改善：更新最佳记录，重置计数器
                best_loss = avg_loss
                patience_counter = 0
                # 保存当前最佳模型（以便后续恢复）
                best_path = os.path.join(config.checkpoint_dir, "sft_best.pt")
                torch.save(model.state_dict(), best_path)
            else:
                # 无明显改善：增加计数器
                patience_counter += 1
                print(f"  ⚠️ Loss 改善不明显 ({patience_counter}/{patience})")

            # 条件3：连续 patience 次无改善，触发早停
            if patience_counter >= patience:
                print(f"\n🛑 早停触发！连续 {patience} 个 epoch 没有明显改善")
                print(f"   最佳 Loss: {best_loss:.4f}")
                # 恢复到最佳模型（避免使用过拟合的最后一个版本）
                best_path = os.path.join(config.checkpoint_dir, "sft_best.pt")
                if os.path.exists(best_path):
                    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
                break

    # 保存模型
    save_path = os.path.join(config.checkpoint_dir, "sft_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"SFT 模型已保存: {save_path}")

    return history


def train_reward_model(
    base_model: GPT,
    tokenizer: BPETokenizer,
    config: argparse.Namespace,
    device: torch.device
):
    """
    阶段 3：训练奖励模型 (Reward Model)

    目标：学习人类偏好，能够给回答打分

    奖励模型的作用：
    ---------------
    在 RLHF 中，我们需要一个"裁判"来评价模型生成的回答好不好。
    这个裁判就是奖励模型。

    工作原理：
    ---------
    1. 输入：(prompt + response) 的完整对话
    2. 输出：一个标量分数，表示回答的质量
    3. 训练数据：人类标注的偏好对 (chosen > rejected)

    训练流程：
    ---------
    1. 从 SFT 模型初始化（共享 Transformer 权重）
    2. 替换输出层为标量奖励头
    3. 使用 Bradley-Terry 损失训练

    参数:
        base_model: 基础 GPT 模型（用于初始化）
        tokenizer: 分词器
        config: 训练配置
        device: 计算设备

    返回:
        训练好的奖励模型，如果没有数据则返回 None
    """
    print("\n" + "=" * 60)
    print("阶段 3：训练奖励模型 (Reward Model)")
    print("=" * 60)

    # ============================================================
    # Step 1: 加载偏好数据
    # ============================================================
    reward_data = load_reward_data()
    if not reward_data:
        print("没有奖励数据，跳过")
        return None  # 返回 None 表示跳过

    # ============================================================
    # Step 2: 导入奖励模型相关类
    # ============================================================
    # 延迟导入避免循环依赖
    from reward_model import RewardModel, RewardModelTrainer

    # ============================================================
    # Step 3: 创建模型配置
    # ============================================================
    # 奖励模型架构与基础模型相同（共享 Transformer 结构）
    model_config = MyLLMConfig(
        vocab_size=len(tokenizer.vocab),  # 词表大小
        emb_dim=config.emb_dim,           # 嵌入维度
        num_heads=config.num_heads,       # 注意力头数
        num_layers=config.num_layers,     # Transformer 层数
        context_size=config.context_size, # 上下文长度
        dropout=config.dropout            # Dropout 比例
    )

    # ============================================================
    # Step 4: 从预训练模型初始化奖励模型
    # ============================================================
    # 这样可以利用预训练学到的语言知识，加速奖励模型训练
    reward_model = RewardModel.from_pretrained(base_model, model_config)

    # ============================================================
    # Step 5: 创建训练器并训练
    # ============================================================
    trainer = RewardModelTrainer(
        reward_model,
        tokenizer,
        model_config,
        learning_rate=config.reward_lr,      # 奖励模型学习率
        num_epochs=config.reward_epochs      # 训练轮数
    )

    # 开始训练
    trainer.train(reward_data, batch_size=config.reward_batch_size)

    # ============================================================
    # Step 6: 保存模型
    # ============================================================
    save_path = os.path.join(config.checkpoint_dir, "reward_model.pt")
    trainer.save_model(save_path)

    return reward_model


def train_rlhf(
    model: GPT,
    reward_model,
    tokenizer: BPETokenizer,
    config: argparse.Namespace,
    device: torch.device
) -> Dict:
    """
    阶段 4：RLHF (PPO) 训练

    目标：利用奖励模型指导策略优化，让模型生成更符合人类偏好的回答

    RLHF 是什么？
    ------------
    RLHF = Reinforcement Learning from Human Feedback
    即"基于人类反馈的强化学习"

    核心思想：
    ---------
    1. 让模型生成回答（采样）
    2. 用奖励模型给回答打分（获取奖励）
    3. 根据分数优化模型（PPO 算法）
    4. 重复以上步骤

    PPO 算法简介：
    -------------
    PPO = Proximal Policy Optimization（近端策略优化）
    - 限制每次更新的幅度，防止策略剧烈变化
    - 使用 clip 机制确保稳定性
    - 是目前 RLHF 中最常用的 RL 算法

    训练循环：
    ---------
    for each episode:
        1. 随机选择一个 prompt
        2. 让模型生成 response
        3. 用奖励模型计算 reward
        4. 计算 PPO 损失并更新模型

    参数:
        model: 策略模型（要优化的 GPT 模型）
        reward_model: 奖励模型（评分器）
        tokenizer: 分词器
        config: 训练配置
        device: 计算设备

    返回:
        训练历史记录
    """
    print("\n" + "=" * 60)
    print("阶段 4：RLHF (PPO) 训练")
    print("=" * 60)

    # ============================================================
    # Step 1: 检查奖励模型是否可用
    # ============================================================
    if reward_model is None:
        print("没有奖励模型，跳过 RLHF")
        return {}  # 没有奖励模型无法进行 RLHF

    # ============================================================
    # Step 2: 导入 PPO 训练器
    # ============================================================
    from rlhf import PPOTrainer, RLHFConfig

    # ============================================================
    # Step 3: 获取训练提示（prompts）
    # ============================================================
    # RLHF 需要 prompt 来引导模型生成回答
    # 我们从 SFT 数据中提取 user 问题作为 prompt
    sft_data = load_sft_data()
    if not sft_data:
        print("没有 SFT 数据提供提示，跳过 RLHF")
        return {}

    # 提取所有用户问题作为 prompts
    prompts = [item['user'] for item in sft_data]

    # ============================================================
    # Step 4: 创建模型配置
    # ============================================================
    model_config = MyLLMConfig(
        vocab_size=len(tokenizer.vocab),
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_size=config.context_size,
        dropout=0.0  # 推理/生成时不使用 dropout（确保一致性）
    )

    # ============================================================
    # Step 5: 配置 RLHF 超参数
    # ============================================================
    rlhf_config = RLHFConfig(
        clip_ratio=0.2,              # PPO 裁剪比例：限制策略更新幅度
        kl_coef=0.01,                # KL 散度系数：惩罚偏离原策略太远
        learning_rate=config.rlhf_lr,  # 学习率
        num_episodes=config.rlhf_episodes,  # 训练轮数
        batch_size=config.rlhf_batch_size,  # 批次大小
        max_new_tokens=64            # 生成的最大 token 数
    )

    # ============================================================
    # Step 6: 创建 PPO 训练器
    # ============================================================
    trainer = PPOTrainer(
        policy_model=model,          # 策略模型（要优化的模型）
        reward_model=reward_model,   # 奖励模型（评分器）
        tokenizer=tokenizer,
        config=model_config,
        rlhf_config=rlhf_config
    )

    # ============================================================
    # Step 7: 开始训练
    # ============================================================
    history = trainer.train(prompts)

    # ============================================================
    # Step 8: 保存模型
    # ============================================================
    save_path = os.path.join(config.checkpoint_dir, "rlhf_final.pt")
    trainer.save_model(save_path)

    return history


def train_rlvf(
    model: GPT,
    tokenizer: BPETokenizer,
    config: argparse.Namespace,
    device: torch.device
) -> Dict:
    """
    阶段 5：RLVF 训练

    目标：利用可验证反馈提升精确推理能力

    RLVF 是什么？
    ------------
    RLVF = Reinforcement Learning from Verifiable Feedback
    即"基于可验证反馈的强化学习"

    与 RLHF 的区别：
    ---------------
    | 特性     | RLHF              | RLVF              |
    |----------|-------------------|-------------------|
    | 反馈来源 | 人类偏好/奖励模型  | 自动验证器        |
    | 适用场景 | 开放式回答         | 有标准答案的问题   |
    | 成本     | 需要人类标注       | 自动化，成本低     |
    | 准确性   | 主观，可能有噪声   | 客观，100%准确     |

    RLVF 的优势：
    -------------
    1. 数学题：答案可以自动验证 (2+3=5 ✓)
    2. 代码题：可以运行测试验证
    3. 逻辑题：可以形式化验证
    4. 无需奖励模型，直接使用正确/错误作为奖励

    训练流程：
    ---------
    for each iteration:
        1. 选择一个数学/逻辑问题
        2. 让模型生成多个答案
        3. 验证答案是否正确
        4. 正确的给正奖励，错误的给负奖励
        5. 使用强化学习更新模型

    参数:
        model: 策略模型
        tokenizer: 分词器
        config: 训练配置
        device: 计算设备

    返回:
        训练历史记录
    """
    print("\n" + "=" * 60)
    print("阶段 5：RLVF 训练")
    print("=" * 60)

    # ============================================================
    # Step 1: 加载 RLVF 数据（数学/逻辑问题）
    # ============================================================
    rlvf_data = load_rlvf_data()
    if not rlvf_data:
        print("没有 RLVF 数据，跳过")
        return {}

    # ============================================================
    # Step 2: 导入 RLVF 训练器
    # ============================================================
    from rlvf import RLVFTrainer, RLVFConfig

    # ============================================================
    # Step 3: 创建模型配置
    # ============================================================
    model_config = MyLLMConfig(
        vocab_size=len(tokenizer.vocab),
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_size=config.context_size,
        dropout=0.0  # 生成时关闭 dropout
    )

    # ============================================================
    # Step 4: 配置 RLVF 超参数
    # ============================================================
    rlvf_config = RLVFConfig(
        num_iterations=config.rlvf_iterations,  # 训练迭代次数
        samples_per_task=2,           # 每个问题生成几个答案
        correct_reward=1.0,           # 答对时的奖励
        incorrect_reward=-0.5,        # 答错时的惩罚（负奖励）
        learning_rate=config.rlvf_lr, # 学习率
        max_new_tokens=32             # 生成的最大 token 数（答案通常很短）
    )

    # ============================================================
    # Step 5: 创建 RLVF 训练器
    # ============================================================
    trainer = RLVFTrainer(
        policy_model=model,
        tokenizer=tokenizer,
        config=model_config,
        rlvf_config=rlvf_config
    )

    # ============================================================
    # Step 6: 开始训练
    # ============================================================
    history = trainer.train(rlvf_data, batch_size=config.rlvf_batch_size)

    # ============================================================
    # Step 7: 保存模型
    # ============================================================
    save_path = os.path.join(config.checkpoint_dir, "rlvf_final.pt")
    trainer.save_model(save_path)

    return history


# ==========================================
# 主函数
# ==========================================
# 程序入口点，负责：
# 1. 解析命令行参数
# 2. 初始化分词器和模型
# 3. 按顺序执行 5 个训练阶段
# 4. 保存最终模型

def main():
    """
    主函数：执行完整的 5 阶段训练流程

    5 阶段训练流程：
    ===============
    ┌─────────────────────────────────────────────────────────────┐
    │  阶段 1: Pretrain（预训练）                                  │
    │  ├── 目标：学习语言规律                                      │
    │  ├── 数据：大量无标注文本                                    │
    │  └── 输出：pretrain_final.pt                                │
    ├─────────────────────────────────────────────────────────────┤
    │  阶段 2: SFT（监督微调）                                     │
    │  ├── 目标：学习对话格式和指令遵循                            │
    │  ├── 数据：人工标注的对话数据                                │
    │  └── 输出：sft_final.pt                                     │
    ├─────────────────────────────────────────────────────────────┤
    │  阶段 3: Reward Model（奖励模型训练）                        │
    │  ├── 目标：学习人类偏好，给回答打分                          │
    │  ├── 数据：偏好对比数据 (chosen vs rejected)                 │
    │  └── 输出：reward_model.pt                                  │
    ├─────────────────────────────────────────────────────────────┤
    │  阶段 4: RLHF（基于人类反馈的强化学习）                      │
    │  ├── 目标：让模型生成更符合人类偏好的回答                    │
    │  ├── 方法：PPO 算法 + 奖励模型                               │
    │  └── 输出：rlhf_final.pt                                    │
    ├─────────────────────────────────────────────────────────────┤
    │  阶段 5: RLVF（可验证反馈强化学习）                          │
    │  ├── 目标：提升精确推理能力（数学、逻辑）                    │
    │  ├── 方法：自动验证答案正确性作为奖励                        │
    │  └── 输出：rlvf_final.pt                                    │
    └─────────────────────────────────────────────────────────────┘

    使用示例：
    ---------
    # 完整训练（从头开始）
    python train.py

    # 跳过预训练（使用已有模型）
    python train.py --skip-pretrain

    # 只训练 SFT（跳过其他阶段）
    python train.py --skip-pretrain --skip-reward --skip-rlhf --skip-rlvf

    # 自定义参数
    python train.py --emb_dim 512 --num_layers 6 --pretrain_epochs 20
    """

    # ============================================================
    # Step 1: 创建命令行参数解析器
    # ============================================================
    # argparse 是 Python 标准库，用于解析命令行参数
    parser = argparse.ArgumentParser(description="MyLLM 完整 5 阶段训练")

    # ------------------------------------------------------------
    # 阶段控制参数：决定跳过哪些训练阶段
    # ------------------------------------------------------------
    # action="store_true" 表示：如果提供了这个参数，值为 True；否则为 False
    parser.add_argument("--skip-pretrain", action="store_true", help="跳过预训练阶段")
    parser.add_argument("--skip-sft", action="store_true", help="跳过 SFT 阶段")
    parser.add_argument("--skip-reward", action="store_true", help="跳过奖励模型训练")
    parser.add_argument("--skip-rlhf", action="store_true", help="跳过 RLHF 阶段")
    parser.add_argument("--skip-rlvf", action="store_true", help="跳过 RLVF 阶段")

    # ------------------------------------------------------------
    # 模型架构参数
    # ------------------------------------------------------------
    parser.add_argument("--vocab_size", type=int, default=2000,
                        help="词表大小：模型能识别的不同 token 数量")
    parser.add_argument("--emb_dim", type=int, default=256,
                        help="嵌入维度：每个 token 的向量表示维度")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="注意力头数：多头注意力机制的头数")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Transformer 层数：模型深度")
    parser.add_argument("--context_size", type=int, default=256,
                        help="上下文长度：模型能处理的最大序列长度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比例：随机丢弃神经元的概率，防止过拟合")

    # ------------------------------------------------------------
    # 通用训练参数
    # ------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小：每次更新使用的样本数")
    parser.add_argument("--seq_len", type=int, default=64,
                        help="序列长度：预训练时每个样本的 token 数")

    # ------------------------------------------------------------
    # 预训练参数
    # ------------------------------------------------------------
    parser.add_argument("--pretrain_epochs", type=int, default=10,
                        help="预训练轮数：遍历整个数据集的次数")
    parser.add_argument("--pretrain_lr", type=float, default=3e-4,
                        help="预训练学习率：3e-4 = 0.0003，是常用的初始学习率")

    # ------------------------------------------------------------
    # SFT 参数
    # ------------------------------------------------------------
    # 注意：epoch 过多会导致过拟合！建议 15-30
    parser.add_argument("--sft_epochs", type=int, default=20,
                        help="SFT 训练轮数（建议 15-30，过多会过拟合）")
    parser.add_argument("--sft_lr", type=float, default=5e-5,
                        help="SFT 学习率：比预训练小，避免破坏预训练知识")

    # ------------------------------------------------------------
    # 奖励模型参数
    # ------------------------------------------------------------
    parser.add_argument("--reward_epochs", type=int, default=15,
                        help="奖励模型训练轮数：需要足够轮次学习偏好")
    parser.add_argument("--reward_lr", type=float, default=1e-5,
                        help="奖励模型学习率：较小的学习率确保稳定")
    parser.add_argument("--reward_batch_size", type=int, default=4,
                        help="奖励模型批次大小：偏好对比需要成对数据")

    # ------------------------------------------------------------
    # RLHF 参数
    # ------------------------------------------------------------
    parser.add_argument("--rlhf_episodes", type=int, default=100,
                        help="RLHF 训练轮数：强化学习需要更多迭代")
    parser.add_argument("--rlhf_lr", type=float, default=1e-5,
                        help="RLHF 学习率：RL 需要小学习率保持稳定")
    parser.add_argument("--rlhf_batch_size", type=int, default=4,
                        help="RLHF 批次大小：生成+评估的开销大，批次较小")

    # ------------------------------------------------------------
    # RLVF 参数
    # ------------------------------------------------------------
    parser.add_argument("--rlvf_iterations", type=int, default=60,
                        help="RLVF 迭代次数：每次迭代处理一批问题")
    parser.add_argument("--rlvf_lr", type=float, default=1e-5,
                        help="RLVF 学习率")
    parser.add_argument("--rlvf_batch_size", type=int, default=4,
                        help="RLVF 批次大小")

    # ------------------------------------------------------------
    # 文件路径参数
    # ------------------------------------------------------------
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="检查点目录：模型保存位置")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json",
                        help="词表路径：分词器保存/加载位置")

    # 解析命令行参数，返回 Namespace 对象
    # 可以用 args.参数名 访问各个参数值
    args = parser.parse_args()

    # ============================================================
    # Step 2: 设置计算设备
    # ============================================================
    # torch.cuda.is_available() 检查是否有可用的 GPU
    # 如果有 GPU，使用 "cuda"；否则使用 "cpu"
    # GPU 训练速度通常比 CPU 快 10-100 倍
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ============================================================
    # Step 3: 创建检查点目录
    # ============================================================
    # exist_ok=True 表示如果目录已存在则不报错
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ==========================================
    # Step 4: 准备分词器
    # ==========================================
    # 分词器将文本转换为模型能理解的数字序列
    # 如果已有训练好的分词器，直接加载；否则从语料训练新的
    print("\n" + "=" * 60)
    print("准备分词器")
    print("=" * 60)

    if os.path.exists(args.vocab_path):
        # 加载已有分词器（包含词表和合并规则）
        print(f"加载已有分词器: {args.vocab_path}")
        tokenizer = BPETokenizer.load(args.vocab_path)
    else:
        # 训练新分词器
        print("训练新分词器...")
        corpus = load_pretrain_data()              # 加载训练语料
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)  # 创建分词器
        tokenizer.fit(corpus, verbose=True)        # 训练 BPE 合并规则
        tokenizer.save(args.vocab_path)            # 保存到文件

    print(f"词表大小: {len(tokenizer.vocab)}")

    # 更新 vocab_size 为实际大小（可能与参数不完全一致）
    args.vocab_size = len(tokenizer.vocab)

    # ==========================================
    # Step 5: 创建模型
    # ==========================================
    print("\n" + "=" * 60)
    print("创建模型")
    print("=" * 60)

    # 创建模型配置对象，包含所有超参数
    model_config = GPTConfig(
        vocab_size=args.vocab_size,     # 词表大小
        emb_dim=args.emb_dim,           # 嵌入维度
        num_heads=args.num_heads,       # 注意力头数
        num_layers=args.num_layers,     # Transformer 层数
        context_size=args.context_size, # 上下文长度
        dropout=args.dropout            # Dropout 比例
    )

    # 创建模型并移动到指定设备（GPU/CPU）
    model = GPT(model_config).to(device)

    # 打印模型参数量（:, 添加千位分隔符）
    print(f"模型参数量: {model.get_num_params():,}")

    # ============================================================
    # Step 6: 加载已有检查点（如果跳过某些阶段）
    # ============================================================
    # 如果跳过预训练，尝试加载已有的预训练模型
    if args.skip_pretrain:
        pretrain_path = os.path.join(args.checkpoint_dir, "pretrain_final.pt")
        if os.path.exists(pretrain_path):
            print(f"加载预训练模型: {pretrain_path}")
            # torch.load 加载模型权重
            # map_location 确保在不同设备间兼容
            # weights_only=True 提高安全性，只加载权重
            model.load_state_dict(torch.load(pretrain_path, map_location=device, weights_only=True))

    # 如果跳过 SFT，尝试加载已有的 SFT 模型
    if args.skip_sft:
        sft_path = os.path.join(args.checkpoint_dir, "sft_final.pt")
        if os.path.exists(sft_path):
            print(f"加载 SFT 模型: {sft_path}")
            model.load_state_dict(torch.load(sft_path, map_location=device, weights_only=True))

    # ==========================================
    # Step 7: 开始 5 阶段训练
    # ==========================================
    # 按顺序执行各个训练阶段（可以通过 --skip-xxx 跳过）

    # ------------------------------------------------------------
    # 阶段 1：预训练 (Pretrain)
    # ------------------------------------------------------------
    # 目的：从大量文本中学习语言规律
    # 任务：预测下一个词（语言建模）
    if not args.skip_pretrain:
        train_pretrain(model, tokenizer, args, device)
    else:
        print("\n跳过预训练阶段")

    # ------------------------------------------------------------
    # 阶段 2：监督微调 (SFT)
    # ------------------------------------------------------------
    # 目的：学习对话格式和指令遵循能力
    # 任务：根据用户输入生成助手回答
    if not args.skip_sft:
        train_sft(model, tokenizer, args, device)
    else:
        print("\n跳过 SFT 阶段")

    # ------------------------------------------------------------
    # 阶段 3：奖励模型训练 (Reward Model)
    # ------------------------------------------------------------
    # 目的：学习人类偏好，能够给回答打分
    # 任务：区分好回答和差回答
    reward_model = None  # 初始化为 None，如果跳过或失败则保持 None
    if not args.skip_reward:
        reward_model = train_reward_model(model, tokenizer, args, device)
    else:
        print("\n跳过奖励模型训练")
        # 如果跳过训练但需要进行 RLHF，尝试加载已有的奖励模型
        reward_path = os.path.join(args.checkpoint_dir, "reward_model.pt")
        if os.path.exists(reward_path) and not args.skip_rlhf:
            # 延迟导入
            from reward_model import RewardModel
            # 创建奖励模型配置
            model_config = MyLLMConfig(
                vocab_size=args.vocab_size,
                emb_dim=args.emb_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                context_size=args.context_size
            )
            # 创建并加载奖励模型
            reward_model = RewardModel(model_config)
            reward_model.load_state_dict(torch.load(reward_path, map_location=device, weights_only=True))
            reward_model.to(device)  # 移动到正确的设备
            print(f"加载已有奖励模型: {reward_path}")

    # ------------------------------------------------------------
    # 阶段 4：RLHF (基于人类反馈的强化学习)
    # ------------------------------------------------------------
    # 目的：利用奖励模型优化策略，生成更好的回答
    # 方法：PPO 算法
    if not args.skip_rlhf:
        train_rlhf(model, reward_model, tokenizer, args, device)
    else:
        print("\n跳过 RLHF 阶段")

    # ------------------------------------------------------------
    # 阶段 5：RLVF (可验证反馈强化学习)
    # ------------------------------------------------------------
    # 目的：提升精确推理能力（数学、逻辑题）
    # 方法：使用自动验证器代替奖励模型
    if not args.skip_rlvf:
        train_rlvf(model, tokenizer, args, device)
    else:
        print("\n跳过 RLVF 阶段")

    # ==========================================
    # Step 8: 训练完成，打印总结
    # ==========================================
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"\n模型文件保存在: {args.checkpoint_dir}/")
    print("  - pretrain_final.pt  (预训练模型)")
    print("  - sft_final.pt       (SFT 模型)")
    print("  - reward_model.pt    (奖励模型)")
    print("  - rlhf_final.pt      (RLHF 模型)")
    print("  - rlvf_final.pt      (RLVF 模型)")


# ============================================================
# 程序入口
# ============================================================
# Python 的标准入口点模式：
# 当直接运行此文件时（python train.py），__name__ == "__main__"
# 当被其他文件 import 时，__name__ == "train"
# 这样可以防止 import 时意外执行训练
if __name__ == "__main__":
    main()
