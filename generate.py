"""
大语言模型推理/生成脚本

功能：
1. 加载训练好的模型
2. 文本生成（支持多种采样策略）
3. 交互式对话模式

生成策略：
- Greedy: 每次选择概率最高的词
- Top-k: 从概率最高的 k 个词中随机选择
- Top-p (Nucleus): 从累积概率达到 p 的词中随机选择
- Temperature: 控制输出的随机性

使用方法：
    python generate.py --prompt "你好" --max_length 50
    python generate.py --interactive  # 进入交互模式
"""

import argparse
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import BPETokenizer


class TextGenerator:
    """
    文本生成器

    支持多种采样策略的温度调节
    """

    def __init__(self, model: GPT, tokenizer: BPETokenizer, device: torch.device):
        """
        Args:
            model: 训练好的 GPT 模型
            tokenizer: 分词器
            device: 运行设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()  # 设置为评估模式

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int = None,
        show_progress: bool = False
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示
            max_length: 最大生成长度（包括 prompt）
            temperature: 温度参数，控制随机性
                - < 1.0: 更保守，输出更确定
                - = 1.0: 标准采样
                - > 1.0: 更随机，输出更多样
            top_k: Top-k 采样，0 表示不使用
            top_p: Top-p (nucleus) 采样，1.0 表示不使用
            eos_token_id: 结束 token ID，遇到则停止
            show_progress: 是否显示生成进度

        Returns:
            生成的文本
        """
        # 1. 编码 prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # 限制生成长度
        max_new_tokens = max_length - len(input_ids)
        if max_new_tokens <= 0:
            return prompt

        generated = input_ids[:]
        if show_progress:
            print(f"提示词: {prompt}")
            print("生成中: ", end="", flush=True)

        # 2. 逐个生成 token
        for _ in range(max_new_tokens):
            # 获取当前序列
            input_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)

            # 截断到最大上下文长度
            if input_tensor.size(1) > self.model.config.context_size:
                input_tensor = input_tensor[:, -self.model.config.context_size:]

            # 前向传播
            logits, _ = self.model(input_tensor)

            # 只取最后一个 token 的 logits
            logits = logits[0, -1, :]  # [vocab_size]

            # 3. 应用温度
            logits = logits / temperature

            # 4. Top-k 过滤
            if top_k > 0:
                # 找到第 k 大的值
                kth_value = torch.topk(logits, top_k).values[-1]
                # 将小于该值的设为 -inf
                logits = torch.where(
                    logits < kth_value,
                    torch.tensor(float('-inf'), device=self.device),
                    logits
                )

            # 5. Top-p (Nucleus) 过滤
            if top_p < 1.0:
                # 按概率排序
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # 计算累积概率
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 找到累积概率超过 top_p 的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                # 移除第一个（保留概率最高的）
                sorted_indices_to_remove[0] = False
                # 将要移除的位置设为 -inf
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')

            # 6. 采样下一个 token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # 检查是否遇到结束符
            if eos_token_id is not None and next_token == eos_token_id:
                break

            # 添加到生成序列
            generated.append(next_token)

            if show_progress:
                token_text = self.tokenizer.decode([next_token])
                print(token_text, end="", flush=True)

        # 7. 解码
        output_text = self.tokenizer.decode(generated)

        if show_progress:
            print()

        return output_text

    @torch.no_grad()
    def generate_greedy(self, prompt: str, max_length: int = 100) -> str:
        """
        贪婪解码：每次选择概率最高的词

        优点：输出确定
        缺点：容易陷入重复循环
        """
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        max_new_tokens = max_length - len(input_ids)
        generated = input_ids[:]

        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long, device=self.device)

            if input_tensor.size(1) > self.model.config.context_size:
                input_tensor = input_tensor[:, -self.model.config.context_size:]

            logits, _ = self.model(input_tensor)
            next_token = logits[0, -1, :].argmax().item()

            generated.append(next_token)

        return self.tokenizer.decode(generated)

    @torch.no_grad()
    def get_perplexity(self, text: str) -> float:
        """
        计算文本的困惑度 (Perplexity)

        困惑度是评估语言模型的重要指标
        越低表示模型对文本的预测越准确

        公式：exp(cross_entropy_loss)
        """
        input_ids = self.tokenizer.encode(text)

        # 构造输入和目标
        input_tensor = torch.tensor([input_ids[:-1]], dtype=torch.long, device=self.device)
        target_tensor = torch.tensor([input_ids[1:]], dtype=torch.long, device=self.device)

        # 前向传播
        logits, loss = self.model(input_tensor, target_tensor)

        # 计算困惑度
        perplexity = torch.exp(loss).item()

        return perplexity


def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device) -> tuple:
    """
    加载模型和分词器

    Args:
        checkpoint_path: 模型检查点路径
        tokenizer_path: 分词器路径
        device: 运行设备

    Returns:
        (model, tokenizer)
    """
    print(f"加载分词器: {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"词表大小: {len(tokenizer.vocab)}")

    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从检查点恢复配置
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = GPTConfig(**config_dict)
    else:
        # 使用默认配置
        config = GPTConfig(vocab_size=len(tokenizer.vocab))

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型参数量: {model.get_num_params():,}")

    return model, tokenizer


def interactive_mode(generator: TextGenerator):
    """
    交互式对话模式
    """
    print("\n" + "=" * 50)
    print("进入交互模式（输入 'quit' 退出）")
    print("=" * 50)

    print("\n可用的命令:")
    print("  直接输入文本 -> 生成回复")
    print("  --temp <0.1-2.0> -> 设置温度")
    print("  --topk <k> -> 设置 top-k")
    print("  --topp <p> -> 设置 top-p")
    print("  --max <n> -> 设置最大长度")
    print("  quit -> 退出")

    temperature = 0.8
    top_k = 10
    top_p = 0.9
    max_length = 100

    while True:
        print("\n" + "-" * 50)
        user_input = input("你: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！")
            break

        # 处理命令
        if user_input.startswith('--temp'):
            try:
                temperature = float(user_input.split()[1])
                print(f"温度已设置为: {temperature}")
                continue
            except:
                print("温度格式错误，使用: --temp <0.1-2.0>")
                continue

        if user_input.startswith('--topk'):
            try:
                top_k = int(user_input.split()[1])
                print(f"Top-k 已设置为: {top_k}")
                continue
            except:
                print("Top-k 格式错误，使用: --topk <k>")
                continue

        if user_input.startswith('--topp'):
            try:
                top_p = float(user_input.split()[1])
                print(f"Top-p 已设置为: {top_p}")
                continue
            except:
                print("Top-p 格式错误，使用: --topp <p>")
                continue

        if user_input.startswith('--max'):
            try:
                max_length = int(user_input.split()[1])
                print(f"最大长度已设置为: {max_length}")
                continue
            except:
                print("最大长度格式错误，使用: --max <n>")
                continue

        # 生成回复
        print("\nAI: ", end="", flush=True)
        response = generator.generate(
            prompt=user_input,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            show_progress=False
        )
        print(response)


def main():
    parser = argparse.ArgumentParser(description="GPT 文本生成")

    # 模型参数
    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt",
                        help="模型检查点路径")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                        help="分词器路径")

    # 生成参数
    parser.add_argument("--prompt", type=str, default="你好",
                        help="输入提示")
    parser.add_argument("--max_length", type=int, default=100,
                        help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="温度参数（0.1-2.0）")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k 采样（0表示不使用）")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p 采样（1.0表示不使用）")
    parser.add_argument("--greedy", action="store_true",
                        help="使用贪婪解码")

    # 模式
    parser.add_argument("--interactive", action="store_true",
                        help="交互式对话模式")
    parser.add_argument("--perplexity", type=str, default="",
                        help="计算指定文本的困惑度")

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)
    generator = TextGenerator(model, tokenizer, device)

    # 计算困惑度
    if args.perplexity:
        text = args.perplexity
        ppl = generator.get_perplexity(text)
        print(f"\n文本: {text}")
        print(f"困惑度: {ppl:.2f}")
        return

    # 交互模式
    if args.interactive:
        interactive_mode(generator)
        return

    # 单次生成
    print("\n" + "=" * 50)
    print("生成配置")
    print("=" * 50)
    print(f"提示词: {args.prompt}")
    print(f"最大长度: {args.max_length}")
    print(f"温度: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print("=" * 50)

    if args.greedy:
        print("\n使用贪婪解码...")
        output = generator.generate_greedy(args.prompt, args.max_length)
    else:
        output = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            show_progress=True
        )

    print("\n" + "=" * 50)
    print("生成结果:")
    print("=" * 50)
    print(output)


if __name__ == "__main__":
    main()
