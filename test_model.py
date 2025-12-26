#!/usr/bin/env python3
"""
模型验证脚本
======================
在训练完成后运行此脚本，验证模型是否正常工作。

使用方法：
    python test_model.py              # 验证默认模型
    python test_model.py --verbose    # 显示详细输出
    python test_model.py --model checkpoints/sft_final.pt  # 指定模型

作者：MyLLM Team
"""

import os
import sys
import argparse
import time
from pathlib import Path


# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")


def print_success(text: str):
    print(f"  {Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str):
    print(f"  {Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text: str):
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text: str):
    print(f"  {Colors.BLUE}ℹ{Colors.RESET} {text}")


def test_model_loading(model_path: str, vocab_path: str, verbose: bool = False) -> tuple:
    """测试模型加载"""
    print("\n[1/5] 测试模型加载...")

    try:
        from inference import MyLLMChat

        start_time = time.time()
        chat = MyLLMChat(model_path, vocab_path)
        load_time = time.time() - start_time

        print_success(f"模型加载成功 (耗时: {load_time:.2f}s)")

        if verbose:
            print_info(f"  模型配置:")
            print_info(f"    - 词表大小: {chat.config.vocab_size}")
            print_info(f"    - 嵌入维度: {chat.config.emb_dim}")
            print_info(f"    - 注意力头数: {chat.config.num_heads}")
            print_info(f"    - 层数: {chat.config.num_layers}")
            print_info(f"    - 上下文长度: {chat.config.context_size}")

        return True, chat

    except FileNotFoundError as e:
        print_error(f"文件不存在: {e}")
        return False, None
    except Exception as e:
        print_error(f"模型加载失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False, None


def test_text_generation(chat, verbose: bool = False) -> bool:
    """测试文本生成（预训练模式）"""
    print("\n[2/5] 测试文本生成（续写模式）...")

    test_prompts = [
        ("人工智能", 30),
        ("今天天气", 20),
        ("学习编程", 25),
    ]

    all_ok = True

    for prompt, max_tokens in test_prompts:
        try:
            start_time = time.time()
            response = chat.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                show_input=True
            )
            gen_time = time.time() - start_time

            # 验证生成结果
            if len(response) > len(prompt):
                print_success(f"'{prompt}' -> 生成 {len(response)-len(prompt)} 字 ({gen_time:.2f}s)")
                if verbose:
                    print_info(f"  输出: {response[:80]}{'...' if len(response) > 80 else ''}")
            else:
                print_warning(f"'{prompt}' -> 未生成新内容")
                all_ok = False

        except Exception as e:
            print_error(f"'{prompt}' -> 生成失败: {e}")
            all_ok = False

    return all_ok


def test_chat_mode(chat, verbose: bool = False) -> bool:
    """测试对话模式"""
    print("\n[3/5] 测试对话模式...")

    test_questions = [
        "你好",
        "你是谁？",
        "什么是人工智能？",
    ]

    all_ok = True
    scores = []

    for question in test_questions:
        try:
            start_time = time.time()
            response = chat.chat(
                question,
                max_new_tokens=80,
                temperature=0.7,
                top_k=50
            )
            gen_time = time.time() - start_time

            # 评估回答质量
            score = evaluate_response(question, response)
            scores.append(score)

            status = "良好" if score >= 0.5 else "一般" if score >= 0.3 else "较差"
            color = Colors.GREEN if score >= 0.5 else Colors.YELLOW if score >= 0.3 else Colors.RED

            print_success(f"Q: '{question}'")
            print_info(f"  A: {response[:60]}{'...' if len(response) > 60 else ''}")
            print_info(f"  评分: {color}{status}{Colors.RESET} ({gen_time:.2f}s)")

            if score < 0.3:
                all_ok = False

        except Exception as e:
            print_error(f"'{question}' -> 对话失败: {e}")
            all_ok = False
            scores.append(0)

    avg_score = sum(scores) / len(scores) if scores else 0
    print_info(f"\n  平均质量评分: {avg_score:.2f}")

    return all_ok


def evaluate_response(question: str, response: str) -> float:
    """
    评估回答质量（简单启发式评分）

    评分标准：
    - 长度合理 (10-200字): +0.3
    - 没有明显重复: +0.3
    - 包含相关词汇: +0.2
    - 语句通顺（无乱码）: +0.2
    """
    score = 0.0

    # 长度检查
    if 5 <= len(response) <= 200:
        score += 0.3
    elif len(response) > 200:
        score += 0.15  # 过长扣分

    # 重复检查
    if len(response) >= 10:
        # 检查是否有连续重复的片段
        has_repetition = False
        for n in [4, 6, 8]:
            for i in range(len(response) - n * 2):
                pattern = response[i:i+n]
                if pattern in response[i+n:]:
                    has_repetition = True
                    break
            if has_repetition:
                break
        if not has_repetition:
            score += 0.3
        else:
            score += 0.1

    # 相关性检查（简单关键词匹配）
    keywords_map = {
        "你好": ["你好", "好", "帮", "问题", "助手"],
        "你是谁": ["我是", "MyLLM", "模型", "语言", "AI", "助手"],
        "人工智能": ["人工智能", "AI", "计算机", "技术", "学习", "智能"],
    }

    relevant_keywords = keywords_map.get(question, [])
    if any(kw in response for kw in relevant_keywords):
        score += 0.2

    # 语句通顺检查（无乱码）
    # 检查是否有连续的非中文非英文字符
    garbage_chars = 0
    for char in response:
        if not ('\u4e00' <= char <= '\u9fff' or  # 中文
                'a' <= char.lower() <= 'z' or     # 英文
                char.isdigit() or                  # 数字
                char in '，。！？、；：""''（）【】《》—…·,.!?;:\'"()-[] \n'):
            garbage_chars += 1

    if garbage_chars / max(len(response), 1) < 0.1:
        score += 0.2

    return min(score, 1.0)


def test_generation_speed(chat, verbose: bool = False) -> bool:
    """测试生成速度"""
    print("\n[4/5] 测试生成速度...")

    try:
        import torch

        prompt = "今天"
        num_tokens = 50
        num_runs = 3

        times = []
        for i in range(num_runs):
            start_time = time.time()
            _ = chat.generate(prompt, max_new_tokens=num_tokens, temperature=0.8)
            times.append(time.time() - start_time)

        avg_time = sum(times) / len(times)
        tokens_per_sec = num_tokens / avg_time

        print_success(f"平均生成速度: {tokens_per_sec:.1f} tokens/秒")
        print_info(f"  生成 {num_tokens} tokens 平均耗时: {avg_time:.2f}s")

        if verbose:
            print_info(f"  各次耗时: {', '.join(f'{t:.2f}s' for t in times)}")

        # 速度评级
        if tokens_per_sec >= 20:
            print_info(f"  速度评级: {Colors.GREEN}快{Colors.RESET}")
        elif tokens_per_sec >= 5:
            print_info(f"  速度评级: {Colors.YELLOW}中等{Colors.RESET}")
        else:
            print_info(f"  速度评级: {Colors.RED}较慢{Colors.RESET}（CPU 训练正常）")

        return True

    except Exception as e:
        print_error(f"速度测试失败: {e}")
        return False


def test_sampling_strategies(chat, verbose: bool = False) -> bool:
    """测试不同采样策略"""
    print("\n[5/5] 测试采样策略...")

    prompt = "人工智能是"

    strategies = [
        {"name": "低温度 (0.3)", "temperature": 0.3, "top_k": 50},
        {"name": "高温度 (1.2)", "temperature": 1.2, "top_k": 50},
        {"name": "Top-k=10", "temperature": 0.8, "top_k": 10},
        {"name": "Top-k=100", "temperature": 0.8, "top_k": 100},
    ]

    all_ok = True

    for strategy in strategies:
        try:
            response = chat.generate(
                prompt,
                max_new_tokens=30,
                temperature=strategy["temperature"],
                top_k=strategy["top_k"],
                show_input=False
            )
            print_success(f"{strategy['name']}: 生成 {len(response)} 字")
            if verbose:
                print_info(f"  输出: {response[:50]}{'...' if len(response) > 50 else ''}")

        except Exception as e:
            print_error(f"{strategy['name']}: 失败 - {e}")
            all_ok = False

    return all_ok


def generate_report(results: dict, model_path: str):
    """生成测试报告"""
    print_header("测试报告")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    print(f"\n模型路径: {model_path}")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n测试结果: {passed}/{total} 通过")
    print("-" * 40)

    for name, status in results.items():
        if status:
            print_success(f"{name}")
        else:
            print_error(f"{name}")

    print("-" * 40)

    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}模型验证通过！可以正常使用。{Colors.RESET}")
        print(f"\n{Colors.BLUE}使用方法:{Colors.RESET}")
        print("  交互对话: python inference.py")
        print("  演示模式: python inference.py demo")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}有 {failed} 项测试未通过。{Colors.RESET}")
        print("模型可能需要更多训练，或检查训练数据质量。")
        print(f"\n{Colors.BLUE}建议:{Colors.RESET}")
        print("  1. 增加训练轮数 (sft_epochs)")
        print("  2. 扩充 SFT 对话数据")
        print("  3. 检查数据格式是否正确")
        return 1


def main():
    parser = argparse.ArgumentParser(description="MyLLM 模型验证工具")
    parser.add_argument(
        "--model",
        default="checkpoints/sft_final.pt",
        help="模型文件路径 (默认: checkpoints/sft_final.pt)"
    )
    parser.add_argument(
        "--vocab",
        default="checkpoints/vocab.json",
        help="词表文件路径 (默认: checkpoints/vocab.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细输出"
    )
    args = parser.parse_args()

    print_header("MyLLM 模型验证工具")
    print(f"模型: {args.model}")
    print(f"词表: {args.vocab}")

    # 检查文件存在
    if not os.path.exists(args.model):
        print_error(f"\n模型文件不存在: {args.model}")
        print_info("请先运行 'python train.py' 训练模型")
        return 1

    if not os.path.exists(args.vocab):
        print_error(f"\n词表文件不存在: {args.vocab}")
        print_info("请先运行 'python train.py' 训练模型")
        return 1

    results = {}

    # 1. 测试模型加载
    load_ok, chat = test_model_loading(args.model, args.vocab, args.verbose)
    results["模型加载"] = load_ok

    if not load_ok:
        print_error("\n模型加载失败，无法继续测试")
        return 1

    # 2. 测试文本生成
    results["文本生成"] = test_text_generation(chat, args.verbose)

    # 3. 测试对话模式
    results["对话模式"] = test_chat_mode(chat, args.verbose)

    # 4. 测试生成速度
    results["生成速度"] = test_generation_speed(chat, args.verbose)

    # 5. 测试采样策略
    results["采样策略"] = test_sampling_strategies(chat, args.verbose)

    # 生成报告
    return generate_report(results, args.model)


if __name__ == "__main__":
    sys.exit(main())
