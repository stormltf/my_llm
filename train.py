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

import os
import argparse
import json
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import GPT, GPTConfig, MyLLM
from config import MyLLMConfig, get_mini_config
from tokenizer import BPETokenizer


# ==========================================
# 数据集类
# ==========================================

class PretrainDataset(Dataset):
    """预训练数据集"""

    def __init__(self, texts: List[str], tokenizer: BPETokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []

        print("正在处理预训练数据...")
        all_token_ids = []
        for text in tqdm(texts, desc="编码文本"):
            token_ids = tokenizer.encode(text)
            all_token_ids.extend(token_ids)

        print(f"总共编码了 {len(all_token_ids)} 个 token")

        for i in range(0, len(all_token_ids) - seq_len - 1):
            input_ids = all_token_ids[i:i + seq_len]
            target_ids = all_token_ids[i + 1:i + seq_len + 1]
            self.samples.append({
                'input_ids': input_ids,
                'target_ids': target_ids
            })

        print(f"生成了 {len(self.samples)} 个训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample['input_ids'], dtype=torch.long),
            torch.tensor(sample['target_ids'], dtype=torch.long)
        )


class SFTDataset(Dataset):
    """SFT 数据集"""

    def __init__(self, data: List[Dict], tokenizer: BPETokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        print("正在处理 SFT 数据...")
        for item in tqdm(data, desc="处理对话"):
            # 构造 ChatML 格式
            text = f"<|im_start|>user\n{item['user']}<|im_end|>\n<|im_start|>assistant\n{item['assistant']}<|im_end|>"
            token_ids = tokenizer.encode(text)

            # 截断
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            # 构造输入和目标（自回归）
            if len(token_ids) > 1:
                input_ids = token_ids[:-1]
                target_ids = token_ids[1:]
                self.samples.append({
                    'input_ids': input_ids,
                    'target_ids': target_ids
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
    """自定义 collate 函数，处理变长序列"""
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # 找到最大长度
    max_len = max(len(ids) for ids in input_ids)

    # Padding
    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(input_ids, target_ids):
        pad_len = max_len - len(inp)
        padded_inputs.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        padded_targets.append(torch.cat([tgt, torch.full((pad_len,), -1, dtype=torch.long)]))

    return torch.stack(padded_inputs), torch.stack(padded_targets)


# ==========================================
# 训练函数
# ==========================================

def load_pretrain_data() -> List[str]:
    """加载预训练数据"""
    data_path = "data/pretrain_data.txt"
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # 使用内置示例数据
        print("未找到预训练数据文件，使用内置示例数据")
        corpus = [
            "我 是 一个 人工智能 助手",
            "人工智能 是 计算机 科学 的 一个 分支",
            "深度 学习 是 机器 学习 的 一种 方法",
            "自然 语言 处理 让 计算机 理解 人类 语言",
            "大 语言 模型 可以 生成 流畅 的 文本",
        ] * 100
        return corpus


def load_sft_data() -> List[Dict]:
    """加载 SFT 数据"""
    data_path = "data/sft_data.json"
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("未找到 SFT 数据文件")
        return []


def load_reward_data() -> List[Dict]:
    """加载奖励模型训练数据"""
    data_path = "data/reward_data.json"
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("未找到奖励数据文件")
        return []


def load_rlvf_data() -> List[Dict]:
    """加载 RLVF 数据"""
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

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.pretrain_lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.pretrain_epochs
    )

    history = {'loss': []}
    model.train()

    for epoch in range(config.pretrain_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch + 1}/{config.pretrain_epochs}")

        for input_ids, target_ids in progress_bar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
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

    for epoch in range(config.sft_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"SFT Epoch {epoch + 1}/{config.sft_epochs}")

        for input_ids, target_ids in progress_bar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        history['loss'].append(avg_loss)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

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
    阶段 3：训练奖励模型

    目标：学习人类偏好，能够给回答打分
    """
    print("\n" + "=" * 60)
    print("阶段 3：训练奖励模型 (Reward Model)")
    print("=" * 60)

    # 加载数据
    reward_data = load_reward_data()
    if not reward_data:
        print("没有奖励数据，跳过")
        return None

    # 导入奖励模型相关类
    from reward_model import RewardModel, RewardModelTrainer

    # 创建模型配置（与基础模型一致）
    model_config = MyLLMConfig(
        vocab_size=len(tokenizer.vocab),
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_size=config.context_size,
        dropout=config.dropout
    )

    # 从预训练模型初始化奖励模型
    reward_model = RewardModel.from_pretrained(base_model, model_config)

    # 训练
    trainer = RewardModelTrainer(
        reward_model,
        tokenizer,
        model_config,
        learning_rate=config.reward_lr,
        num_epochs=config.reward_epochs
    )

    trainer.train(reward_data, batch_size=config.reward_batch_size)

    # 保存模型
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

    目标：利用奖励模型指导策略优化
    """
    print("\n" + "=" * 60)
    print("阶段 4：RLHF (PPO) 训练")
    print("=" * 60)

    if reward_model is None:
        print("没有奖励模型，跳过 RLHF")
        return {}

    # 导入 PPO 训练器
    from rlhf import PPOTrainer, RLHFConfig

    # 从 SFT 数据获取提示
    sft_data = load_sft_data()
    if not sft_data:
        print("没有 SFT 数据提供提示，跳过 RLHF")
        return {}

    prompts = [item['user'] for item in sft_data]

    # 创建模型配置
    model_config = MyLLMConfig(
        vocab_size=len(tokenizer.vocab),
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_size=config.context_size,
        dropout=0.0  # 推理时不使用 dropout
    )

    # RLHF 配置
    rlhf_config = RLHFConfig(
        clip_ratio=0.2,
        kl_coef=0.01,
        learning_rate=config.rlhf_lr,
        num_episodes=config.rlhf_episodes,
        batch_size=config.rlhf_batch_size,
        max_new_tokens=64
    )

    # 创建 PPO 训练器
    trainer = PPOTrainer(
        policy_model=model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=model_config,
        rlhf_config=rlhf_config
    )

    # 训练
    history = trainer.train(prompts)

    # 保存模型
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
    """
    print("\n" + "=" * 60)
    print("阶段 5：RLVF 训练")
    print("=" * 60)

    # 加载数据
    rlvf_data = load_rlvf_data()
    if not rlvf_data:
        print("没有 RLVF 数据，跳过")
        return {}

    # 导入 RLVF 训练器
    from rlvf import RLVFTrainer, RLVFConfig

    # 创建模型配置
    model_config = MyLLMConfig(
        vocab_size=len(tokenizer.vocab),
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_size=config.context_size,
        dropout=0.0
    )

    # RLVF 配置
    rlvf_config = RLVFConfig(
        num_iterations=config.rlvf_iterations,
        samples_per_task=2,
        correct_reward=1.0,
        incorrect_reward=-0.5,
        learning_rate=config.rlvf_lr,
        max_new_tokens=32
    )

    # 创建训练器
    trainer = RLVFTrainer(
        policy_model=model,
        tokenizer=tokenizer,
        config=model_config,
        rlvf_config=rlvf_config
    )

    # 训练
    history = trainer.train(rlvf_data, batch_size=config.rlvf_batch_size)

    # 保存模型
    save_path = os.path.join(config.checkpoint_dir, "rlvf_final.pt")
    trainer.save_model(save_path)

    return history


# ==========================================
# 主函数
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="MyLLM 完整 5 阶段训练")

    # 阶段控制
    parser.add_argument("--skip-pretrain", action="store_true", help="跳过预训练阶段")
    parser.add_argument("--skip-sft", action="store_true", help="跳过 SFT 阶段")
    parser.add_argument("--skip-reward", action="store_true", help="跳过奖励模型训练")
    parser.add_argument("--skip-rlhf", action="store_true", help="跳过 RLHF 阶段")
    parser.add_argument("--skip-rlvf", action="store_true", help="跳过 RLVF 阶段")

    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=6400, help="词表大小")
    parser.add_argument("--emb_dim", type=int, default=256, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer 层数")
    parser.add_argument("--context_size", type=int, default=256, help="上下文长度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比例")

    # 通用训练参数
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--seq_len", type=int, default=64, help="序列长度")

    # 预训练参数
    parser.add_argument("--pretrain_epochs", type=int, default=5, help="预训练轮数")
    parser.add_argument("--pretrain_lr", type=float, default=3e-4, help="预训练学习率")

    # SFT 参数
    parser.add_argument("--sft_epochs", type=int, default=5, help="SFT 训练轮数")
    parser.add_argument("--sft_lr", type=float, default=1e-4, help="SFT 学习率")

    # 奖励模型参数
    parser.add_argument("--reward_epochs", type=int, default=3, help="奖励模型训练轮数")
    parser.add_argument("--reward_lr", type=float, default=1e-5, help="奖励模型学习率")
    parser.add_argument("--reward_batch_size", type=int, default=4, help="奖励模型批次大小")

    # RLHF 参数
    parser.add_argument("--rlhf_episodes", type=int, default=50, help="RLHF 训练轮数")
    parser.add_argument("--rlhf_lr", type=float, default=1e-5, help="RLHF 学习率")
    parser.add_argument("--rlhf_batch_size", type=int, default=4, help="RLHF 批次大小")

    # RLVF 参数
    parser.add_argument("--rlvf_iterations", type=int, default=30, help="RLVF 迭代次数")
    parser.add_argument("--rlvf_lr", type=float, default=1e-5, help="RLVF 学习率")
    parser.add_argument("--rlvf_batch_size", type=int, default=4, help="RLVF 批次大小")

    # 路径
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="检查点目录")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json", help="词表路径")

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ==========================================
    # 准备分词器
    # ==========================================
    print("\n" + "=" * 60)
    print("准备分词器")
    print("=" * 60)

    if os.path.exists(args.vocab_path):
        print(f"加载已有分词器: {args.vocab_path}")
        tokenizer = BPETokenizer.load(args.vocab_path)
    else:
        print("训练新分词器...")
        corpus = load_pretrain_data()
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.fit(corpus, verbose=True)
        tokenizer.save(args.vocab_path)

    print(f"词表大小: {len(tokenizer.vocab)}")

    # 更新 vocab_size 为实际大小
    args.vocab_size = len(tokenizer.vocab)

    # ==========================================
    # 创建模型
    # ==========================================
    print("\n" + "=" * 60)
    print("创建模型")
    print("=" * 60)

    model_config = GPTConfig(
        vocab_size=args.vocab_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        context_size=args.context_size,
        dropout=args.dropout
    )

    model = GPT(model_config).to(device)
    print(f"模型参数量: {model.get_num_params():,}")

    # 如果跳过预训练，尝试加载已有模型
    if args.skip_pretrain:
        pretrain_path = os.path.join(args.checkpoint_dir, "pretrain_final.pt")
        if os.path.exists(pretrain_path):
            print(f"加载预训练模型: {pretrain_path}")
            model.load_state_dict(torch.load(pretrain_path, map_location=device, weights_only=True))

    if args.skip_sft:
        sft_path = os.path.join(args.checkpoint_dir, "sft_final.pt")
        if os.path.exists(sft_path):
            print(f"加载 SFT 模型: {sft_path}")
            model.load_state_dict(torch.load(sft_path, map_location=device, weights_only=True))

    # ==========================================
    # 开始训练
    # ==========================================

    # 阶段 1：预训练
    if not args.skip_pretrain:
        train_pretrain(model, tokenizer, args, device)
    else:
        print("\n跳过预训练阶段")

    # 阶段 2：SFT
    if not args.skip_sft:
        train_sft(model, tokenizer, args, device)
    else:
        print("\n跳过 SFT 阶段")

    # 阶段 3：奖励模型
    reward_model = None
    if not args.skip_reward:
        reward_model = train_reward_model(model, tokenizer, args, device)
    else:
        print("\n跳过奖励模型训练")
        # 尝试加载已有奖励模型
        reward_path = os.path.join(args.checkpoint_dir, "reward_model.pt")
        if os.path.exists(reward_path) and not args.skip_rlhf:
            from reward_model import RewardModel
            model_config = MyLLMConfig(
                vocab_size=args.vocab_size,
                emb_dim=args.emb_dim,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                context_size=args.context_size
            )
            reward_model = RewardModel(model_config)
            reward_model.load_state_dict(torch.load(reward_path, map_location=device, weights_only=True))
            reward_model.to(device)
            print(f"加载已有奖励模型: {reward_path}")

    # 阶段 4：RLHF
    if not args.skip_rlhf:
        train_rlhf(model, reward_model, tokenizer, args, device)
    else:
        print("\n跳过 RLHF 阶段")

    # 阶段 5：RLVF
    if not args.skip_rlvf:
        train_rlvf(model, tokenizer, args, device)
    else:
        print("\n跳过 RLVF 阶段")

    # ==========================================
    # 训练完成
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


if __name__ == "__main__":
    main()
