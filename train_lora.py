"""
LoRA 微调训练脚本

使用 LoRA（Low-Rank Adaptation）对预训练模型进行高效微调。
只训练少量参数（通常 < 1%），大幅降低训练成本。

使用方法：
    # 基础用法
    python train_lora.py

    # 自定义参数
    python train_lora.py --base_model checkpoints/sft_final.pt --epochs 5 --lr 1e-4

    # 修改 LoRA 配置
    python train_lora.py --lora_r 16 --lora_alpha 32 --target_modules q_proj v_proj k_proj o_proj
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import GPT, GPTConfig
from tokenizer import MyLLMTokenizer
from lora import LoRAConfig, LoRATrainer, apply_lora_to_model, save_lora


class SFTDataset(Dataset):
    """
    SFT 数据集，用于 LoRA 微调
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: MyLLMTokenizer,
        max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # 加载数据
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 处理每条对话
        for item in tqdm(data, desc="处理数据"):
            user_text = item.get("user", "")
            assistant_text = item.get("assistant", "")

            # 构建对话格式
            prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            response = f"{assistant_text}<|im_end|>"

            # 编码
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)

            # 合并
            input_ids = prompt_ids + response_ids

            # 截断或填充
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            if len(input_ids) < 2:
                continue

            # 创建目标（向后移一位）
            self.samples.append({
                "input_ids": input_ids[:-1],
                "target_ids": input_ids[1:]
            })

        print(f"加载了 {len(self.samples)} 个训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["input_ids"], dtype=torch.long),
            torch.tensor(sample["target_ids"], dtype=torch.long)
        )


def collate_fn(batch):
    """
    动态填充 batch
    """
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # 找到最长序列
    max_len = max(len(ids) for ids in input_ids)

    # 填充
    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(input_ids, target_ids):
        pad_len = max_len - len(inp)
        padded_inputs.append(
            torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_targets.append(
            torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return torch.stack(padded_inputs), torch.stack(padded_targets)


def load_base_model(
    model_path: str,
    vocab_path: str,
    device: torch.device
) -> GPT:
    """
    加载基础模型
    """
    print(f"加载基础模型: {model_path}")

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)

    # 获取配置
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
    else:
        # 默认配置
        config_dict = {
            "vocab_size": 2000,
            "emb_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "context_size": 256,
            "dropout": 0.1
        }

    config = GPTConfig(**config_dict)
    model = GPT(config)

    # 加载权重
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print(f"模型参数量: {model.get_num_params():,}")

    return model


def train_lora(args):
    """
    LoRA 训练主函数
    """
    print("=" * 60)
    print("LoRA 微调训练")
    print("=" * 60)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载分词器
    print("\n" + "-" * 40)
    print("步骤 1: 加载分词器")
    print("-" * 40)
    tokenizer = MyLLMTokenizer(vocab_path=args.vocab_path)

    # 2. 加载基础模型
    print("\n" + "-" * 40)
    print("步骤 2: 加载基础模型")
    print("-" * 40)
    model = load_base_model(args.base_model, args.vocab_path, device)

    # 3. 配置 LoRA
    print("\n" + "-" * 40)
    print("步骤 3: 配置 LoRA")
    print("-" * 40)
    lora_config = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules
    )

    # 应用 LoRA
    model = apply_lora_to_model(model, lora_config, verbose=True)
    model = model.to(device)

    # 4. 准备数据
    print("\n" + "-" * 40)
    print("步骤 4: 准备数据")
    print("-" * 40)
    dataset = SFTDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 5. 设置优化器
    print("\n" + "-" * 40)
    print("步骤 5: 设置优化器")
    print("-" * 40)

    # 只优化 LoRA 参数
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in lora_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )

    # 6. 训练循环
    print("\n" + "-" * 40)
    print("步骤 6: 开始训练")
    print("-" * 40)

    training_log = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # 前向传播
            _, loss = model(input_ids, target_ids)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  学习率: {current_lr:.6f}")

        # 记录日志
        training_log.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": current_lr,
            "timestamp": datetime.now().isoformat()
        })

        # 保存检查点
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            save_lora(model, checkpoint_dir, lora_config)

    # 7. 保存最终模型
    print("\n" + "-" * 40)
    print("步骤 7: 保存模型")
    print("-" * 40)

    final_dir = os.path.join(args.output_dir, "final")
    save_lora(model, final_dir, lora_config)

    # 保存训练日志
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    print(f"训练日志已保存: {log_path}")

    print("\n" + "=" * 60)
    print("LoRA 训练完成!")
    print("=" * 60)

    return model


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调训练")

    # 模型和数据路径
    parser.add_argument(
        "--base_model", type=str,
        default="checkpoints/sft_final.pt",
        help="基础模型路径"
    )
    parser.add_argument(
        "--vocab_path", type=str,
        default="checkpoints/vocab.json",
        help="词表路径"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="data/sft_data.json",
        help="训练数据路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="checkpoints/lora",
        help="输出目录"
    )

    # LoRA 参数
    parser.add_argument(
        "--lora_r", type=int, default=8,
        help="LoRA 秩"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16,
        help="LoRA 缩放因子"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--target_modules", type=str, nargs="+",
        default=["c_attn", "c_proj"],
        help="要应用 LoRA 的模块（MyLLM: c_attn, c_proj, linear1, linear2）"
    )

    # 训练参数
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="批次大小"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="权重衰减"
    )
    parser.add_argument(
        "--max_length", type=int, default=128,
        help="最大序列长度"
    )
    parser.add_argument(
        "--save_every", type=int, default=1,
        help="每多少轮保存一次"
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 60)
    print("LoRA 训练配置")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)

    train_lora(args)


if __name__ == "__main__":
    main()
