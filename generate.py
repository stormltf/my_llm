"""
大语言模型推理/生成脚本

功能：
1. 加载训练好的模型
2. 自动检测并加载 LoRA 权重（如果存在）
3. 文本生成（支持多种采样策略）
4. 交互式对话模式

生成策略：
- Greedy: 每次选择概率最高的词
- Top-k: 从概率最高的 k 个词中随机选择
- Top-p (Nucleus): 从累积概率达到 p 的词中随机选择
- Temperature: 控制输出的随机性

使用方法：
    python generate.py --prompt "你好" --max_length 50
    python generate.py --interactive  # 进入交互模式
    python generate.py --interactive --lora checkpoints/lora/final  # 指定 LoRA 路径
"""

import argparse
import os
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

        # 获取模型词表大小，用于裁剪 token ID
        self.vocab_size = model.config.vocab_size
        # 如果分词器词表比模型词表大，需要裁剪
        if len(tokenizer.vocab) > self.vocab_size:
            print(f"注意: 分词器词表({len(tokenizer.vocab)}) > 模型词表({self.vocab_size})，将裁剪 token ID")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: int = None,
        repetition_penalty: float = 1.2,
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
            repetition_penalty: 重复惩罚系数，>1.0 降低已生成 token 的概率
            show_progress: 是否显示生成进度

        Returns:
            生成的文本
        """
        # 1. 编码 prompt
        input_ids = self.tokenizer.encode(prompt)
        # 裁剪 token ID 到模型词表大小范围内
        input_ids = [min(tid, self.vocab_size - 1) for tid in input_ids]
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

            # 3.5 应用重复惩罚
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty

            # ============================================================
            # 4. Top-k 过滤
            # ============================================================
            # Top-k 采样：只从概率最高的 k 个 token 中选择
            #
            # 工作原理：
            #   1. 找到第 k 大的 logit 值
            #   2. 将所有小于该值的 logit 设为 -inf
            #   3. softmax 后这些位置的概率变为 0
            #
            # 举例（假设 top_k=3，词表有 5 个词）：
            #   原始 logits:  [2.5, 1.0, 3.0, 0.5, 2.0]
            #   排序后:       [3.0, 2.5, 2.0, 1.0, 0.5]
            #   第 k 大值:    2.0 (第 3 大)
            #   过滤后:       [2.5, -inf, 3.0, -inf, 2.0]
            #   softmax 后:   [0.24, 0, 0.66, 0, 0.10]
            #                 只有 top-3 的词有非零概率
            if top_k > 0:
                # topk 返回 (values, indices)，取第 k 大的值（即 values 的最后一个）
                kth_value = torch.topk(logits, top_k).values[-1]
                # 将小于该值的设为 -inf
                logits = torch.where(
                    logits < kth_value,
                    torch.tensor(float('-inf'), device=self.device),
                    logits
                )

            # ============================================================
            # 5. Top-p (Nucleus) 采样
            # ============================================================
            # Top-p 采样：选择累积概率达到 p 的最小 token 集合
            #
            # 为什么使用 Top-p 而不是 Top-k？
            # ------------------------------
            # Top-k 的问题：固定选择 k 个词，但不同上下文的概率分布差异很大
            #   - 确定性上下文："北京是中国的首" → "都" 概率可能 > 90%
            #     此时 top-k=50 会引入太多噪声
            #   - 不确定上下文："我喜欢吃" → 多个词概率相近
            #     此时 top-k=3 可能太少
            #
            # Top-p 的优势：自适应调整候选词数量
            #   - 分布集中时（高置信度）：候选词少
            #   - 分布分散时（不确定）：候选词多
            #
            # 算法步骤详解（以 top_p=0.9 为例）：
            # ┌────────────────────────────────────────────────────────┐
            # │ Step 1: 按概率降序排列                                  │
            # │   原始 probs:    [0.05, 0.30, 0.10, 0.40, 0.15]        │
            # │   排序后 probs:  [0.40, 0.30, 0.15, 0.10, 0.05]        │
            # │   排序后 indices: [3,    1,    4,    2,    0]          │
            # │                                                        │
            # │ Step 2: 计算累积概率                                    │
            # │   cumsum:        [0.40, 0.70, 0.85, 0.95, 1.00]        │
            # │                                                        │
            # │ Step 3: 找到超过阈值的位置                               │
            # │   > 0.9?:        [F,    F,    F,    T,    T]           │
            # │                  保留─────────────→│移除───────→        │
            # │                                                        │
            # │ Step 4: 保留第一个（即使累积已超过 p）                   │
            # │   为什么？因为至少要保留一个 token 可以采样               │
            # │                                                        │
            # │ Step 5: 将被移除的 token 的 logits 设为 -inf            │
            # │   原始 indices 2 和 0 对应的位置被 mask                 │
            # └────────────────────────────────────────────────────────┘
            if top_p < 1.0:
                # Step 1: 按 logits 降序排序
                # sorted_indices 记录原始位置（后续用于映射回去）
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)

                # Step 2: 计算累积概率
                # 先 softmax 转为概率，再 cumsum 累加
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Step 3: 标记需要移除的位置（累积概率超过 top_p）
                sorted_indices_to_remove = cumulative_probs > top_p

                # Step 4: 始终保留概率最高的 token（索引 0）
                # 这是关键！即使单个 token 概率已超过 top_p，也要保留它
                # 否则可能出现没有可用 token 的情况
                # 举例：如果 top_p=0.3，但最高概率的词已经有 0.5 的概率
                #       cumulative_probs[0] = 0.5 > 0.3，会被标记为移除
                #       但我们至少需要保留一个词，所以强制保留第一个
                sorted_indices_to_remove[0] = False

                # Step 5: 将要移除的 token 的 logits 设为 -inf
                # 使用排序后的索引映射回原始位置
                # sorted_indices[sorted_indices_to_remove] 获取需要移除的原始索引
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


def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device,
               lora_path: str = None) -> tuple:
    """
    加载模型和分词器

    自动检测并加载 LoRA 权重（如果存在）

    Args:
        checkpoint_path: 模型检查点路径
        tokenizer_path: 分词器路径
        device: 运行设备
        lora_path: LoRA 权重路径（None 时自动检测）

    Returns:
        (model, tokenizer)
    """
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 获取 state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # ============================================================
    # 先从权重推断 vocab_size
    # ============================================================
    model_vocab_size = None
    for key in state_dict.keys():
        if 'token_embedding.weight' in key or 'tok_emb.weight' in key:
            model_vocab_size = state_dict[key].shape[0]
            print(f"从权重推断模型 vocab_size: {model_vocab_size}")
            break

    # ============================================================
    # 加载分词器
    # ============================================================
    print(f"加载分词器: {tokenizer_path}")

    # 检查词表文件是否是 BPE 格式
    import json
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            # BPE 格式包含 vocab, merges 等字段
            is_bpe_format = isinstance(vocab_data, dict) and 'vocab' in vocab_data
    except:
        is_bpe_format = False

    if is_bpe_format:
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"BPE 分词器，词表大小: {len(tokenizer.vocab)}")
    else:
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"词表大小: {len(tokenizer.vocab)}")

    # 检查词表大小是否匹配
    tokenizer_vocab_size = len(tokenizer.vocab)
    if model_vocab_size and tokenizer_vocab_size != model_vocab_size:
        print(f"警告: 分词器词表({tokenizer_vocab_size}) 与模型词表({model_vocab_size}) 不匹配")
        print(f"将截取分词器到模型词表大小")

    # ============================================================
    # 检测模型类型并推断配置
    # ============================================================
    # 检查键名格式来判断模型类型
    # MyLLM 使用 "transformer_blocks"，GPT 使用 "blocks"
    has_transformer_blocks = any('transformer_blocks.' in k for k in state_dict.keys())
    has_blocks = any('blocks.' in k for k in state_dict.keys())

    # 从检查点恢复配置，或从权重推断
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = GPTConfig(**config_dict)
        model = GPT(config).to(device)
    else:
        # 从权重推断配置
        config_dict = {}

        # 使用之前推断的 vocab_size
        if model_vocab_size:
            config_dict["vocab_size"] = model_vocab_size

        # 从 token_embedding/tok_emb.weight 推断 emb_dim
        for key in state_dict.keys():
            if 'token_embedding.weight' in key or 'tok_emb.weight' in key:
                config_dict["emb_dim"] = state_dict[key].shape[1]
                break

        # 从 position_embedding/pos_emb.weight 推断 context_size
        for key in state_dict.keys():
            if 'position_embedding.weight' in key or 'pos_emb.weight' in key:
                config_dict["context_size"] = state_dict[key].shape[0]
                print(f"从权重推断 context_size: {config_dict['context_size']}")
                break

        # 从权重推断层数
        if has_transformer_blocks:
            num_layers = max(
                int(k.split('.')[1])
                for k in state_dict.keys()
                if 'transformer_blocks.' in k and '.ln_1' in k
            ) + 1
            config_dict["num_layers"] = num_layers
        elif has_blocks:
            num_layers = max(
                int(k.split('.')[1])
                for k in state_dict.keys()
                if 'blocks.' in k and '.ln_1' in k
            ) + 1
            config_dict["num_layers"] = num_layers

        # 设置默认值
        config_dict.setdefault("vocab_size", model_vocab_size or tokenizer_vocab_size)
        config_dict.setdefault("emb_dim", 256)
        config_dict.setdefault("num_heads", 4)
        config_dict.setdefault("num_layers", 6)
        config_dict.setdefault("context_size", 256)
        config_dict.setdefault("dropout", 0.1)

        config = GPTConfig(**config_dict)
        model = GPT(config).to(device)

    # 加载权重（处理键名兼容性）
    # 检查是否需要键名转换（旧模型使用 transformer_blocks/token_embedding）
    needs_conversion = any('transformer_blocks' in k or 'token_embedding' in k for k in state_dict.keys())

    if needs_conversion:
        print("检测到旧键名格式，正在转换...")
        # 转换旧键名到新键名
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # 转换键名映射
            new_key = new_key.replace('transformer_blocks', 'blocks')
            new_key = new_key.replace('token_embedding', 'tok_emb')
            new_key = new_key.replace('position_embedding', 'pos_emb')
            new_key = new_key.replace('final_norm', 'ln_f')
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        print("键名转换完成")

    # 加载权重（strict=False 因为 model.py 中的别名导致 state_dict 包含多套键名）
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"模型参数量: {model.get_num_params():,}")

    # ============================================================
    # 自动检测并加载 LoRA 权重
    # ============================================================
    # 1. 检查是否指定了 lora_path
    # 2. 否则检查默认路径 checkpoints/lora/final
    # 3. 如果存在，自动加载
    lora_loaded = False

    if lora_path is None:
        # 自动检测默认 LoRA 路径
        default_lora_paths = [
            "checkpoints/lora/final",
            "checkpoints/lora/epoch_3",
            "checkpoints/lora/epoch_2",
            "checkpoints/lora/epoch_1",
        ]
        for path in default_lora_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "lora_weights.pt")):
                lora_path = path
                break

    # 加载 LoRA（如果找到）
    if lora_path and os.path.exists(lora_path):
        lora_weights_path = os.path.join(lora_path, "lora_weights.pt")
        lora_config_path = os.path.join(lora_path, "lora_config.json")

        if os.path.exists(lora_weights_path) and os.path.exists(lora_config_path):
            try:
                from lora import LoRAConfig, apply_lora_to_model, load_lora_state_dict
                import json

                print(f"\n检测到 LoRA 权重: {lora_path}")

                # 加载 LoRA 配置
                with open(lora_config_path, "r") as f:
                    lora_config_dict = json.load(f)
                lora_config = LoRAConfig(**lora_config_dict)

                # 应用 LoRA
                print(f"应用 LoRA 配置: r={lora_config.r}, alpha={lora_config.alpha}")
                print(f"目标模块: {lora_config.target_modules}")
                model = apply_lora_to_model(model, lora_config, verbose=False)

                # 加载 LoRA 权重
                lora_state_dict = torch.load(lora_weights_path, map_location=device, weights_only=True)
                load_lora_state_dict(model, lora_state_dict)

                # 统计 LoRA 参数
                lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())

                print(f"LoRA 权重已加载")
                print(f"  LoRA 可训练参数: {lora_params:,}")
                print(f"  模型总参数: {total_params:,}")
                lora_loaded = True

            except ImportError as e:
                print(f"警告: 无法导入 lora 模块: {e}")
            except Exception as e:
                print(f"警告: 加载 LoRA 失败: {e}")

    if not lora_loaded:
        print("\n使用基础模型（未加载 LoRA）")

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

        # 跳过空输入
        if not user_input:
            continue

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
    parser.add_argument("--checkpoint", type=str, default="checkpoints/sft_final.pt",
                        help="模型检查点路径")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/vocab.json",
                        help="分词器路径")
    parser.add_argument("--lora", type=str, default=None,
                        help="LoRA 权重路径（不指定则自动检测）")

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

    # 加载模型（自动检测并加载 LoRA）
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device, args.lora)
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
