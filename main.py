"""
大语言模型学习项目 - 主入口

这个项目演示了从零构建一个 GPT 风格的大语言模型的完整流程。

项目结构：
    tokenizer.py      - BPE 分词器实现
    model.py          - GPT 模型组件实现
    train.py          - 训练脚本
    generate.py       - 文本生成/推理脚本
    main.py           - 本文件，项目入口和示例

使用流程：
    1. 运行 python main.py --mode demo    - 查看所有组件演示
    2. 运行 python main.py --mode train   - 训练模型
    3. 运行 python main.py --mode generate - 生成文本
    4. 运行 python main.py --mode interactive - 交互式对话

"""

import argparse
import os
import torch

from model import GPT, GPTConfig, demo_model
from tokenizer import BPETokenizer, demo_tokenizer


def demo_all():
    """
    运行所有组件的演示
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "大语言模型学习项目演示")
    print("=" * 70)

    print("\n【第一步：分词器演示】")
    print("-" * 70)
    demo_tokenizer()

    print("\n【第二步：模型组件演示】")
    print("-" * 70)
    demo_model()

    print("\n【第三步：完整流程演示】")
    print("-" * 70)
    demo_full_pipeline()

    print("\n" + "=" * 70)
    print("演示完成！现在可以运行以下命令：")
    print("  python main.py --mode train       # 训练模型")
    print("  python main.py --mode generate    # 生成文本")
    print("  python main.py --mode interactive # 交互式对话")
    print("=" * 70)


def demo_full_pipeline():
    """
    完整流程演示：从数据准备到模型训练，再到文本生成

    这是学习 LLM 最核心的流程，展示了：
    1. 数据准备：文本 → token IDs
    2. 模型训练：学习"预测下一个词"
    3. 文本生成：逐个生成 token，自回归生成
    """

    # ============================================================
    # 步骤 0: 设置随机种子（保证每次运行结果一致，便于调试）
    # ============================================================
    torch.manual_seed(42)  # 42 是"生命、宇宙、一切"的答案，来自银河系漫游指南

    # ============================================================
    # 步骤 1: 准备训练数据
    # ============================================================
    print("\n1. 准备训练数据...")
    print("   原始语料（中文用空格分词）:")

    # 定义简单的训练语料
    # 注意：这里用空格分隔是因为我们的 BPE 实现是按空格分词的
    # 实际项目中，英文天然有空格，中文需要先分词（如 jieba）
    corpus = [
        "我 喜欢 人工智能",
        "人工智能 是 未来",
        "我 喜欢 学习",
        "学习 有趣",
        "你 好",
        "你 好 我 是 助手",
    ] * 20  # 重复 20 次增加数据量，总共 6 * 20 = 120 条句子

    # ============================================================
    # 步骤 2: 训练分词器（Tokenizer）
    # ============================================================
    print("2. 训练 BPE 分词器...")
    print("   BPE 工作原理：从字符开始，逐步合并高频字符对")

    # 创建分词器，词表大小设为 100（足够演示用）
    # 词表越大，表达能力越强，但训练也越慢
    tokenizer = BPETokenizer(vocab_size=100)

    # fit() 方法会：
    # 1. 统计所有字符，建立初始词表
    # 2. 反复合并最高频的相邻字符对，直到词表达到指定大小
    tokenizer.fit(corpus, verbose=False)  # verbose=False 不打印详细过程

    print(f"   词表大小: {len(tokenizer.vocab)} 个 token")
    print(f"   示例: '{corpus[0]}' 编码后 = {tokenizer.encode(corpus[0])}")

    # ============================================================
    # 步骤 3: 创建 GPT 模型
    # ============================================================
    print("\n3. 创建 GPT 模型...")

    # 配置模型参数
    # 这里用一个小型配置，方便快速演示
    config = GPTConfig(
        vocab_size=len(tokenizer.vocab),  # 词表大小必须和分词器一致
        emb_dim=64,                       # 嵌入维度：每个 token 用 64 维向量表示
                                         # GPT-3 用的是 12288，我们用 64 足够演示
        num_heads=2,                      # 注意力头数：2 个头并行处理
                                         # 必须能被 emb_dim 整除：64 / 2 = 32 维/头
        num_layers=2,                     # Transformer 层数：2 层
                                         # GPT-3 有 96 层，我们用 2 层足够演示
        context_size=64,                  # 最大上下文长度：最多看 64 个 token
        dropout=0.0,                      # Dropout 比例：0 表示不用
                                         # 小模型容易欠拟合，不需要 dropout
    )

    # 创建模型实例
    model = GPT(config)

    # 打印模型参数量（数字越大，模型越"聪明"，但也越慢）
    print(f"   模型参数量: {model.get_num_params():,}")
    print(f"   对比: GPT-3 有 1750 亿参数，我们只有 {model.get_num_params() / 1e6:.1f} 万")

    # ============================================================
    # 步骤 4: 准备训练数据
    # ============================================================
    print("\n4. 准备训练样本...")

    # 4.1 将所有文本编码成 token ID 序列
    # 例如：["我 喜欢", "人工智能"] → [4, 24, 27]
    all_token_ids = []
    for text in corpus:
        all_token_ids.extend(tokenizer.encode(text))

    print(f"   总 Token 数: {len(all_token_ids)}")

    # 4.2 构造训练样本
    #
    # 关键概念：语言模型的训练目标是"预测下一个词"
    #
    # 例如：序列是 [我, 喜欢, 学习, 人工智能]
    #       输入 x: [我,  喜欢, 学习]      → 预测下一个是 "学习"
    #       输入 x: [我,  喜欢, 学习, 人工智能] → 预测下一个是 "人工智能"
    #
    # 实际上我们是并行训练的：
    #   输入: [我, 喜欢, 学习]       目标: [喜欢, 学习, 人工智能]
    #   每个位置都预测它后面的那个词

    seq_len = 16  # 训练序列长度，每次取 16 个连续 token 作为输入

    samples = []
    # 滑动窗口构造样本
    for i in range(len(all_token_ids) - seq_len - 1):
        # 输入：从 i 开始的 seq_len 个 token
        input_ids = all_token_ids[i:i + seq_len]
        # 目标：从 i+1 开始的 seq_len 个 token（就是输入整体向后移一位）
        target_ids = all_token_ids[i + 1:i + seq_len + 1]

        samples.append((input_ids, target_ids))

    print(f"   训练样本数: {len(samples)}")
    print(f"   样本示例:")
    print(f"     输入: {samples[0][0]}")
    print(f"     目标: {samples[0][1]} (输入整体向后移一位)")

    # ============================================================
    # 步骤 5: 训练模型
    # ============================================================
    print("\n5. 开始训练模型...")

    # 5.1 创建优化器
    # AdamW 是目前训练 LLM 最常用的优化器
    # lr (learning rate): 学习率，控制每次参数更新的步长
    #    - 太大：可能跳过最优解
    #    - 太小：收敛太慢
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)  # lr=0.01

    print("   训练配置:")
    print(f"     优化器: AdamW")
    print(f"     学习率: 0.01")
    print(f"     训练轮数: 50")

    # 5.2 训练循环
    model.train()  # 设置为训练模式（启用 dropout 等）

    for epoch in range(50):  # 训练 50 轮
        total_loss = 0

        # 遍历所有训练样本
        for input_ids, target_ids in samples:
            # 将列表转换为 PyTorch 张量
            # 形状: [seq_len] → [1, seq_len] (增加 batch 维度)
            input_tensor = torch.tensor([input_ids])
            target_tensor = torch.tensor([target_ids])

            # ===== 前向传播 =====
            # input_tensor: [1, 16] 输入序列
            # target_tensor: [1, 16] 目标序列
            # logits: [1, 16, vocab_size] 模型对每个位置预测的词概率
            # loss: 标量，预测与目标之间的差距
            logits, loss = model(input_tensor, target_tensor)

            # ===== 反向传播 =====
            # 1. 清空之前的梯度
            optimizer.zero_grad()

            # 2. 计算梯度（求导）
            # loss.backward() 会计算 loss 对每个参数的偏导数
            # 告诉我们：调整哪个参数能让 loss 变小
            loss.backward()

            # 3. 更新参数
            # optimizer.step() 根据梯度调整参数
            optimizer.step()

            # 累加 loss
            total_loss += loss.item()

        # 计算平均 loss
        avg_loss = total_loss / len(samples)

        # 每 10 轮打印一次
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/50, Loss: {avg_loss:.4f}")

    print("   训练完成！")

    # ============================================================
    # 步骤 6: 文本生成（推理）
    # ============================================================
    print("\n6. 文本生成（推理）...")

    model.eval()  # 设置为评估模式（禁用 dropout 等）

    # 测试提示词
    test_prompts = ["我", "你", "人工智能"]

    for prompt in test_prompts:
        print(f"\n   提示词: '{prompt}'")

        # 6.1 编码提示词
        # 例如："我" → [4]
        input_ids = tokenizer.encode(prompt)
        print(f"   编码结果: {input_ids}")

        # 6.2 自回归生成
        # 核心思想：每次生成一个 token，然后把它加入输入，继续生成下一个
        #
        # 第1轮: 输入 [我] → 预测下一个是 [喜欢]
        # 第2轮: 输入 [我, 喜欢] → 预测下一个是 [学习]
        # 第3轮: 输入 [我, 喜欢, 学习] → 预测下一个是 [人工智能]
        # ...循环...

        generated = input_ids[:]  # 复制一份，避免修改原数据

        for step in range(10):  # 生成 10 个 token
            # 将当前序列转换为张量，形状 [1, seq_len]
            input_tensor = torch.tensor([generated])

            # 前向传播，获取模型输出
            # logits 形状: [1, seq_len, vocab_size]
            logits, _ = model(input_tensor)

            # 取最后一个位置的输出（预测下一个词）
            # logits[0, -1, :] 形状: [vocab_size]
            # 每个词一个分数（logit），分数越大表示概率越高
            next_token_logits = logits[0, -1, :]

            # 贪婪解码：选择分数最大的词
            # argmax() 返回最大值的索引，即预测的 token ID
            next_token = next_token_logits.argmax().item()

            # 将预测的 token 加入序列
            generated.append(next_token)

        # 6.3 解码生成的 token 序列
        output = tokenizer.decode(generated)
        print(f"   生成结果: '{output}'")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 50)
    print("核心要点总结：")
    print("=" * 50)
    print("1. 训练目标：预测下一个词")
    print("2. 训练方式：输入 [x1,x2,x3]，目标 [x2,x3,x4]")
    print("3. 生成方式：逐个生成，每次预测一个词后加入输入")
    print("4. 损失下降 = 模型在变好（预测更准确）")
    print("=" * 50)


def train_model(args):
    """
    训练模型的主入口函数

    功能：
    1. 加载或训练分词器
    2. 准备训练数据集
    3. 创建 GPT 模型
    4. 运行训练循环
    5. 定期保存检查点

    参数：
        args: 命令行参数对象，包含模型配置和训练配置
    """
    from train import train

    # ============================================================
    # 构建训练配置类
    # ============================================================
    # 将命令行参数组装成训练脚本需要的配置对象
    class TrainConfig:
        def __init__(self):
            # 模型参数
            self.vocab_size = args.vocab_size
            self.emb_dim = args.emb_dim
            self.num_heads = args.num_heads
            self.num_layers = args.num_layers
            self.context_size = args.context_size
            self.dropout = args.dropout

            # 训练参数
            self.epochs = args.epochs
            self.batch_size = args.batch_size
            self.lr = args.lr
            self.weight_decay = args.weight_decay
            self.grad_clip = args.grad_clip
            self.seq_len = args.seq_len

            # 保存和加载
            self.checkpoint_dir = args.checkpoint_dir
            self.log_dir = args.log_dir
            self.tokenizer_path = args.tokenizer_path
            self.checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_epoch_100.pt")
            self.save_every = args.save_every
            self.resume = args.resume

    config = TrainConfig()
    train(config)


def generate_text(args):
    """
    文本生成函数

    功能：
    1. 加载训练好的模型和分词器
    2. 使用指定的提示词生成文本
    3. 支持温度、top-k、top-p 等采样策略

    参数说明：
        temperature: 控制输出随机性
            - < 1.0: 更保守，输出更确定
            - = 1.0: 标准采样
            - > 1.0: 更随机，输出更多样
        top_k: 只从概率最高的 k 个词中选择
        top_p: 从累积概率达到 p 的词中选择（nucleus sampling）
    """
    from generate import load_model, TextGenerator

    # ============================================================
    # 1. 设置设备（GPU/CPU）
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用 CPU")

    # ============================================================
    # 2. 加载模型和分词器
    # ============================================================
    checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pt")

    # 检查模型是否存在
    if not os.path.exists(checkpoint_path):
        print(f"\n错误: 模型文件不存在")
        print(f"  路径: {checkpoint_path}")
        print(f"\n请先运行训练命令:")
        print(f"  python main.py --mode train")
        return

    print(f"\n正在加载模型...")
    model, tokenizer = load_model(checkpoint_path, args.tokenizer_path, device)
    generator = TextGenerator(model, tokenizer, device)

    # ============================================================
    # 3. 生成文本
    # ============================================================
    print(f"\n" + "=" * 50)
    print(f"生成配置")
    print(f"=" * 50)
    print(f"  提示词: {args.prompt}")
    print(f"  最大长度: {args.max_length}")
    print(f"  温度: {args.temperature} (越小越确定)")
    print(f"  Top-k: {args.top_k}")
    print(f"  Top-p: {args.top_p}")
    print(f"=" * 50)

    print(f"\n生成中:\n")

    output = generator.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        show_progress=True
    )

    print(f"\n" + "=" * 50)
    print(f"生成结果")
    print(f"=" * 50)
    print(f"{output}")
    print(f"=" * 50)


def interactive_mode(args):
    """
    交互式对话模式

    功能：
    1. 加载训练好的模型
    2. 进入 REPL 循环，等待用户输入
    3. 根据用户输入生成回复
    4. 支持动态调整生成参数（温度、top-k、top-p 等）

    交互命令：
        直接输入文本 → 生成回复
        --temp <0.1-2.0> → 调整温度
        --topk <k> → 调整 top-k
        --topp <p> → 调整 top-p
        --max <n> → 调整最大长度
        quit / exit / q → 退出
    """
    from generate import load_model, TextGenerator

    # ============================================================
    # 1. 设置设备（GPU/CPU）
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================================================
    # 2. 加载模型
    # ============================================================
    checkpoint_path = os.path.join(args.checkpoint_dir, "final_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"\n错误: 模型文件不存在")
        print(f"  路径: {checkpoint_path}")
        print(f"\n请先运行训练命令:")
        print(f"  python main.py --mode train")
        return

    print(f"\n正在加载模型...")
    model, tokenizer = load_model(checkpoint_path, args.tokenizer_path, device)
    generator = TextGenerator(model, tokenizer, device)

    # ============================================================
    # 3. 进入交互循环
    # ============================================================
    from generate import interactive_mode as run_interactive
    run_interactive(generator)


def main():
    """
    主入口函数

    功能：
    1. 解析命令行参数
    2. 根据指定的模式调用相应的功能函数

    使用示例：
        python main.py --mode demo                    # 运行演示
        python main.py --mode train                   # 训练模型
        python main.py --mode train --epochs 50       # 自定义训练轮数
        python main.py --mode generate --prompt "你好" # 生成文本
        python main.py --mode interactive             # 交互模式
    """
    # ============================================================
    # 1. 创建参数解析器
    # ============================================================
    parser = argparse.ArgumentParser(
        description="大语言模型学习项目 - 从零实现 GPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --mode demo
  python main.py --mode train --epochs 50 --lr 0.001
  python main.py --mode generate --prompt "你好" --temperature 0.8
  python main.py --mode interactive
        """
    )

    # ============================================================
    # 2. 定义命令行参数
    # ============================================================

    # ----- 运行模式 -----
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "train", "generate", "interactive"],
                        help="运行模式: demo(演示) | train(训练) | generate(生成) | interactive(交互)")

    # ----- 模型结构参数 -----
    parser.add_argument("--vocab_size", type=int, default=1000,
                        help="词表大小，取决于分词器 (默认: 1000)")
    parser.add_argument("--emb_dim", type=int, default=256,
                        help="词嵌入维度，越大表达能力越强但越慢 (默认: 256)")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="多头注意力的头数，必须能被 emb_dim 整除 (默认: 8)")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Transformer 层数，越多模型越深 (默认: 6)")
    parser.add_argument("--context_size", type=int, default=256,
                        help="最大上下文长度，模型一次能看多少个 token (默认: 256)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比例，防止过拟合 (默认: 0.1)")

    # ----- 训练参数 -----
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数，遍历整个数据集的次数 (默认: 10，适合小数据集)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小，每次训练用多少样本 (默认: 32)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率，控制参数更新步长 (默认: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减，L2 正则化系数 (默认: 0.01)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="梯度裁剪阈值，防止梯度爆炸 (默认: 1.0)")
    parser.add_argument("--seq_len", type=int, default=64,
                        help="训练序列长度 (默认: 64)")
    parser.add_argument("--save_every", type=int, default=10,
                        help="每多少轮保存一次检查点 (默认: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="从检查点恢复训练")

    # ----- 生成参数 -----
    parser.add_argument("--prompt", type=str, default="你好",
                        help="输入提示词 (默认: '你好')")
    parser.add_argument("--max_length", type=int, default=100,
                        help="最大生成长度 (默认: 100)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="温度参数，0.1=保守, 1.0=标准, 2.0=随机 (默认: 0.8)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k 采样，0=不使用 (默认: 10)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) 采样，1.0=不使用 (默认: 0.9)")

    # ----- 路径参数 -----
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="模型检查点保存目录 (默认: checkpoints)")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="训练日志保存目录 (默认: logs)")
    parser.add_argument("--tokenizer_path", type=str, default="data/tokenizer.json",
                        help="分词器文件路径 (默认: data/tokenizer.json)")

    # ============================================================
    # 3. 解析参数
    # ============================================================
    args = parser.parse_args()

    # ============================================================
    # 4. 根据模式执行对应功能
    # ============================================================
    if args.mode == "demo":
        # 演示模式：展示完整流程
        demo_all()

    elif args.mode == "train":
        # 训练模式：训练模型
        train_model(args)

    elif args.mode == "generate":
        # 生成模式：生成文本
        generate_text(args)

    elif args.mode == "interactive":
        # 交互模式：对话式生成
        interactive_mode(args)


# ============================================================
# 程序入口
# ============================================================
# 当直接运行此文件时，执行 main() 函数
# 当被其他文件 import 时，不执行
if __name__ == "__main__":
    main()
