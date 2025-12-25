"""
大语言模型训练脚本

功能：
1. 数据加载和预处理
2. 训练循环
3. 损失监控和模型保存
4. 支持断点续训

使用方法：
    python train.py --epochs 100 --batch_size 32 --lr 1e-3
"""

import os
import argparse
import json
from datetime import datetime
from typing import List, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


class TextDataset(Dataset):
    """
    文本数据集

    将长文本切分成固定长度的序列，用于训练
    """

    def __init__(self, texts: List[str], tokenizer: BPETokenizer, seq_len: int):
        """
        Args:
            texts: 文本列表
            tokenizer: 分词器
            seq_len: 序列长度
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.samples = []

        # 对所有文本进行编码
        print("正在处理训练数据...")
        all_token_ids = []
        for text in tqdm(texts, desc="编码文本"):
            token_ids = tokenizer.encode(text)
            all_token_ids.extend(token_ids)

        print(f"总共编码了 {len(all_token_ids)} 个 token")

        # 切分成固定长度的样本
        # 每个样本包含 seq_len 个输入 token 和 seq_len 个目标 token
        # 目标就是输入向后移一位（预测下一个词）
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


def create_sample_data():
    """
    创建简单的中文训练数据

    在实际项目中，这里应该加载大规模语料
    """
    # 这里使用简单的中文句子作为示例
    # 实际训练时应该使用大规模语料库（如维基百科、Common Crawl 等）
    corpus = [
        # 介绍句
        "我 是 一个 人工智能 助手",
        "我 喜欢 学习 机器 学习 和 深度 学习",
        "自然 语言 处理 是 人工智能 的 重要 分支",
        "大模型 可以 理解 和 生成 人类 语言",

        # 对话模式
        "你 好 我 是 人工智能 助手",
        "你 好 请 问 有 什么 可以 帮助 你",
        "我 想 了解 人工智能",
        "人工智能 是 研究 如何 让 机器 具有 智能 的 学科",

        # 问答模式
        "什么 是 深度 学习",
        "深度 学习 是 机器 学习 的 一种 方法",
        "深度 学习 使用 神经 网络 进行 学习",
        "神经 网络 是 受 人脑 启发 的 算法",

        # 技术相关
        "Python 是 一种 编程 语言",
        "PyTorch 是 一个 深度 学习 框架",
        "我们 使用 PyTorch 构建 大模型",
        "大模型 需要 大量 数据 进行 训练",

        # 重复训练数据以增加样本量
        "我 可以 帮助 你 学习 编程",
        "编程 需要 逻辑 思维 和 实践",
        "学习 编程 要 多 写 代码",
        "代码 是 程序员 的 语言",

        # 更多样本
        "今天 天气 真好",
        "我 喜欢 晴天 的 天气",
        "学习 是 一件 有趣 的 事情",
        "坚持 学习 可以 让 你 变得 更 强",

        # 扩展
        "人工智能 可以 应用 于 许多 领域",
        "医疗 教育 金融 交通 都 有 人工智能 的 应用",
        "未来 人工智能 会 改变 我们 的 生活",
        "我们 应该 拥抱 技术 进步",

        # 更多重复
        "你 好 我 是 人工智能 助手",
        "我 喜欢 学习 机器 学习 和 深度 学习",
        "自然 语言 处理 是 人工智能 的 重要 分支",
        "大模型 可以 理解 和 生成 人类 语言",
        "我 可以 帮助 你 学习 编程",
        "编程 需要 逻辑 思维 和 实践",
        "学习 编程 要 多 写 代码",
        "代码 是 程序员 的 语言",

        # 更多
        "你好 今天 天气 真好",
        "我 喜欢 晴朗 的 天气",
        "学习 是 一件 非常 有趣 的 事情",
        "坚持 学习 可以 让 你 不断 进步",

        # 更多对话
        "请 问 你 什么 是 人工智能",
        "人工智能 是 模拟 人类 智能 的 技术",
        "人工智能 可以 学习 推理 和 解决 问题",
        "人工智能 的 应用 越来越 广泛",

        # 更多扩展
        "深度 学习 需要 大量 数据 和 算力",
        "GPU 是 深度 学习 的 重要 工具",
        "我们 使用 GPU 来 加速 模型 训练",
        "训练 一个 大模型 需要 很 长 时间",
    ]

    # 重复数据以增加训练量
    corpus = corpus * 50  # 重复50次

    return corpus


@torch.no_grad()
def estimate_loss(model: GPT, data_loader: DataLoader, device: torch.device):
    """
    估算模型在数据集上的损失

    使用 no_grad() 上下文，不计算梯度，节省内存
    """
    model.eval()
    total_loss = 0
    count = 0

    for input_ids, target_ids in data_loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        _, loss = model(input_ids, target_ids)
        total_loss += loss.item() * input_ids.size(0)
        count += input_ids.size(0)

    model.train()
    return total_loss / count


def train(config: argparse.Namespace):
    """
    训练主函数
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 1. 准备分词器
    print("\n" + "=" * 60)
    print("步骤 1: 准备分词器")
    print("=" * 60)

    if os.path.exists(config.tokenizer_path):
        # 加载已有分词器
        print(f"加载已有分词器: {config.tokenizer_path}")
        tokenizer = BPETokenizer.load(config.tokenizer_path)
    else:
        # 训练新分词器
        print("训练新分词器...")
        corpus = create_sample_data()
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
        tokenizer.fit(corpus, verbose=True)
        tokenizer.save(config.tokenizer_path)

    print(f"词表大小: {len(tokenizer.vocab)}")

    # 2. 准备数据
    print("\n" + "=" * 60)
    print("步骤 2: 准备训练数据")
    print("=" * 60)

    corpus = create_sample_data()
    print(f"训练语料包含 {len(corpus)} 个句子")

    # 创建数据集
    # 注意：seq_len 需要小于等于模型的 context_size
    seq_len = min(config.seq_len, 128)  # 示例中限制为128
    dataset = TextDataset(corpus, tokenizer, seq_len=seq_len)
    print(f"数据集大小: {len(dataset)}")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    # 3. 创建模型
    print("\n" + "=" * 60)
    print("步骤 3: 创建模型")
    print("=" * 60)

    model_config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        emb_dim=config.emb_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_size=config.context_size,
        dropout=config.dropout,
    )

    model = GPT(model_config).to(device)
    print(f"模型参数量: {model.get_num_params():,}")

    # 4. 优化器
    print("\n" + "=" * 60)
    print("步骤 4: 设置优化器")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay
    )
    print(f"优化器: AdamW, 学习率: {config.lr}")

    # 学习率调度器（可选）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.lr * 0.1
    )

    # 5. 训练循环
    print("\n" + "=" * 60)
    print("步骤 5: 开始训练")
    print("=" * 60)

    start_epoch = 0
    global_step = 0

    # 断点续训
    if config.resume and os.path.exists(config.checkpoint_path):
        print(f"从检查点恢复训练: {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        print(f"从第 {start_epoch} 轮继续训练")

    # 训练日志
    log_file = os.path.join(config.log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")

        for input_ids, target_ids in progress_bar:
            # 移动数据到设备
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # 前向传播
            _, loss = model(input_ids, target_ids)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            # 更新参数
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()
            global_step += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)

        # 打印训练信息
        print(f"\nEpoch {epoch + 1}/{config.epochs} 完成")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存日志
        log_entry = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'timestamp': datetime.now().isoformat()
        }
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        # 保存检查点
        if (epoch + 1) % config.save_every == 0 or epoch == config.epochs - 1:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config.__dict__,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"  检查点已保存: {checkpoint_path}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    # 保存最终模型
    final_path = os.path.join(config.checkpoint_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config.__dict__,
    }, final_path)
    print(f"最终模型已保存: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="训练 GPT 模型")

    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=1000, help="词表大小")
    parser.add_argument("--emb_dim", type=int, default=256, help="嵌入维度")
    parser.add_argument("--num_heads", type=int, default=8, help="注意力头数")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer 层数")
    parser.add_argument("--context_size", type=int, default=256, help="最大上下文长度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 比例")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--seq_len", type=int, default=64, help="训练序列长度")

    # 保存和加载
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="检查点目录")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json", help="分词器路径")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/checkpoint_epoch_100.pt", help="检查点文件路径")
    parser.add_argument("--save_every", type=int, default=10, help="每多少轮保存一次")
    parser.add_argument("--resume", action="store_true", help="从检查点恢复训练")

    args = parser.parse_args()

    # 打印配置
    print("=" * 60)
    print("训练配置")
    print("=" * 60)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 60)

    train(args)


if __name__ == "__main__":
    main()
