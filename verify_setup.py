#!/usr/bin/env python3
"""
环境和数据验证脚本
======================
在训练前运行此脚本，确保环境配置正确、数据文件完整。

使用方法：
    python verify_setup.py

作者：MyLLM Team
"""

import os
import sys
import json
from pathlib import Path


# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")


def print_success(text: str):
    """打印成功信息"""
    print(f"  {Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str):
    """打印错误信息"""
    print(f"  {Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text: str):
    """打印普通信息"""
    print(f"  {Colors.BLUE}ℹ{Colors.RESET} {text}")


def check_python_version() -> bool:
    """检查 Python 版本"""
    print("\n[1/6] 检查 Python 版本...")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major >= 3 and version.minor >= 8:
        print_success(f"Python 版本: {version_str} (>= 3.8 ✓)")
        return True
    else:
        print_error(f"Python 版本: {version_str} (需要 >= 3.8)")
        return False


def check_dependencies() -> bool:
    """检查依赖包"""
    print("\n[2/6] 检查依赖包...")

    all_ok = True

    # 检查 torch
    try:
        import torch
        print_success(f"PyTorch 版本: {torch.__version__}")

        # 检查 CUDA
        if torch.cuda.is_available():
            print_success(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print_success("MPS 可用 (Apple Silicon 加速)")
        else:
            print_warning("GPU 不可用，将使用 CPU 训练（速度较慢）")
    except ImportError:
        print_error("PyTorch 未安装")
        print_info("  安装命令: pip install torch")
        all_ok = False

    # 检查 numpy
    try:
        import numpy as np
        print_success(f"NumPy 版本: {np.__version__}")
    except ImportError:
        print_error("NumPy 未安装")
        print_info("  安装命令: pip install numpy")
        all_ok = False

    return all_ok


def check_project_files() -> bool:
    """检查项目核心文件"""
    print("\n[3/6] 检查项目文件...")

    required_files = [
        ("config.py", "模型配置"),
        ("tokenizer.py", "分词器"),
        ("model.py", "模型定义"),
        ("train.py", "训练脚本"),
        ("inference.py", "推理脚本"),
    ]

    all_ok = True
    base_path = Path(__file__).parent

    for filename, description in required_files:
        filepath = base_path / filename
        if filepath.exists():
            print_success(f"{filename} - {description}")
        else:
            print_error(f"{filename} 缺失 - {description}")
            all_ok = False

    return all_ok


def check_data_files() -> tuple:
    """检查数据文件"""
    print("\n[4/6] 检查数据文件...")

    base_path = Path(__file__).parent
    data_dir = base_path / "data"
    all_ok = True
    warnings = []

    # 检查数据目录
    if not data_dir.exists():
        print_error("data/ 目录不存在")
        print_info("  请创建 data/ 目录并添加训练数据")
        return False, []

    # 检查预训练数据
    pretrain_file = data_dir / "pretrain_data.txt"
    if pretrain_file.exists():
        with open(pretrain_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = [l for l in content.split('\n') if l.strip()]
            char_count = len(content)

        print_success(f"pretrain_data.txt - {len(lines)} 行, {char_count} 字符")

        if char_count < 1000:
            warnings.append("预训练数据量较少，建议增加更多文本以获得更好效果")
    else:
        print_warning("pretrain_data.txt 不存在（将使用内置数据）")

    # 检查 SFT 数据
    sft_file = data_dir / "sft_data.json"
    if sft_file.exists():
        try:
            with open(sft_file, 'r', encoding='utf-8') as f:
                sft_data = json.load(f)

            if not isinstance(sft_data, list):
                print_error("sft_data.json 格式错误：应该是数组")
                all_ok = False
            else:
                print_success(f"sft_data.json - {len(sft_data)} 条对话")

                # 验证数据格式
                valid_count = 0
                for i, item in enumerate(sft_data):
                    if isinstance(item, dict) and 'user' in item and 'assistant' in item:
                        valid_count += 1
                    else:
                        print_warning(f"  第 {i+1} 条数据格式不正确，应包含 'user' 和 'assistant' 字段")

                if valid_count < len(sft_data):
                    print_warning(f"  有效数据: {valid_count}/{len(sft_data)}")

                if len(sft_data) < 20:
                    warnings.append("SFT 对话数据较少，建议增加更多对话以提升效果")

        except json.JSONDecodeError as e:
            print_error(f"sft_data.json 解析失败: {e}")
            all_ok = False
    else:
        print_warning("sft_data.json 不存在（将使用内置数据）")

    return all_ok, warnings


def check_checkpoints() -> bool:
    """检查已有的模型检查点"""
    print("\n[5/6] 检查模型检查点...")

    base_path = Path(__file__).parent
    checkpoint_dir = base_path / "checkpoints"

    if not checkpoint_dir.exists():
        print_info("checkpoints/ 目录不存在（训练时会自动创建）")
        return True

    # 检查是否有预训练模型
    model_files = list(checkpoint_dir.glob("*.pt"))
    if model_files:
        print_success(f"发现 {len(model_files)} 个模型文件:")
        for mf in model_files:
            size_mb = mf.stat().st_size / (1024 * 1024)
            print_info(f"  {mf.name} ({size_mb:.1f} MB)")
    else:
        print_info("暂无模型文件（训练后会生成）")

    # 检查词表
    vocab_file = checkpoint_dir / "vocab.json"
    if vocab_file.exists():
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print_success(f"vocab.json - 词表大小: {len(vocab.get('token_to_id', {}))} 个 token")
    else:
        print_info("vocab.json 不存在（训练时会自动生成）")

    return True


def check_imports() -> bool:
    """测试项目模块导入"""
    print("\n[6/6] 测试模块导入...")

    all_ok = True

    try:
        from config import MyLLMConfig, get_mini_config
        print_success("config.py 导入成功")
    except Exception as e:
        print_error(f"config.py 导入失败: {e}")
        all_ok = False

    try:
        from tokenizer import MyLLMTokenizer
        print_success("tokenizer.py 导入成功")
    except Exception as e:
        print_error(f"tokenizer.py 导入失败: {e}")
        all_ok = False

    try:
        from model import MyLLM
        print_success("model.py 导入成功")
    except Exception as e:
        print_error(f"model.py 导入失败: {e}")
        all_ok = False

    return all_ok


def run_quick_test() -> bool:
    """运行快速功能测试"""
    print("\n[附加] 快速功能测试...")

    try:
        from config import get_mini_config
        from tokenizer import MyLLMTokenizer
        from model import MyLLM
        import torch

        # 创建配置
        config = get_mini_config()
        print_success("创建配置成功")

        # 创建分词器并构建词表
        tokenizer = MyLLMTokenizer()
        tokenizer.build_vocab("测试文本，用于验证分词器功能。", max_vocab_size=100)
        print_success(f"分词器初始化成功 (词表大小: {tokenizer.vocab_size})")

        # 测试编码解码
        test_text = "你好"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        print_success(f"编码解码测试: '{test_text}' -> {encoded} -> '{decoded}'")

        # 创建模型
        config.vocab_size = tokenizer.vocab_size
        model = MyLLM(config)
        param_count = sum(p.numel() for p in model.parameters())
        print_success(f"模型创建成功 (参数量: {param_count/1e6:.2f}M)")

        # 测试前向传播
        input_ids = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            logits, _ = model(input_ids)
        print_success(f"前向传播测试成功 (输出形状: {logits.shape})")

        return True

    except Exception as e:
        print_error(f"功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print_header("MyLLM 环境验证工具")
    print("此工具将检查您的环境配置和数据文件是否正确。\n")

    results = []
    warnings = []

    # 运行所有检查
    results.append(("Python 版本", check_python_version()))
    results.append(("依赖包", check_dependencies()))
    results.append(("项目文件", check_project_files()))

    data_ok, data_warnings = check_data_files()
    results.append(("数据文件", data_ok))
    warnings.extend(data_warnings)

    results.append(("检查点", check_checkpoints()))
    results.append(("模块导入", check_imports()))

    # 如果基础检查都通过，运行功能测试
    all_basic_ok = all(r[1] for r in results)
    if all_basic_ok:
        results.append(("功能测试", run_quick_test()))

    # 汇总结果
    print_header("验证结果汇总")

    passed = 0
    failed = 0

    for name, status in results:
        if status:
            print_success(f"{name}: 通过")
            passed += 1
        else:
            print_error(f"{name}: 失败")
            failed += 1

    # 显示警告
    if warnings:
        print(f"\n{Colors.YELLOW}警告信息:{Colors.RESET}")
        for w in warnings:
            print_warning(w)

    # 最终结论
    print("\n" + "-" * 60)
    if failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}所有检查通过！环境配置正确。{Colors.RESET}")
        print(f"\n{Colors.BLUE}下一步:{Colors.RESET}")
        print("  1. 运行训练: python train.py")
        print("  2. 训练后验证: python test_model.py")
        print("  3. 开始对话: python inference.py")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}有 {failed} 项检查未通过，请根据上述提示修复问题。{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
