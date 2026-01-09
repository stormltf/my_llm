#!/usr/bin/env python3
"""
MyLLM 测试运行脚本

使用方法：
    python run_tests.py              # 运行所有测试
    python run_tests.py -v           # 详细输出
    python run_tests.py --fast       # 只运行快速测试
    python run_tests.py --module tokenizer  # 只测试分词器
    python run_tests.py --coverage   # 生成覆盖率报告

作者：MyLLM Team
"""

import os
import sys
import argparse
import subprocess
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
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str):
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


def check_dependencies():
    """检查测试依赖"""
    print_info("检查测试依赖...")

    required = ['pytest', 'torch']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print_success(f"{pkg} 已安装")
        except ImportError:
            missing.append(pkg)
            print_error(f"{pkg} 未安装")

    if missing:
        print(f"\n{Colors.YELLOW}请安装缺失的依赖：{Colors.RESET}")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def run_tests(args):
    """运行测试"""
    print_header("MyLLM 测试套件")

    if not check_dependencies():
        return 1

    # 构建 pytest 命令
    cmd = [sys.executable, '-m', 'pytest']

    # 测试目录
    test_dir = Path(__file__).parent / 'tests'

    # 根据参数选择测试
    if args.module:
        module_map = {
            'tokenizer': 'test_tokenizer.py',
            'model': 'test_model.py',
            'config': 'test_config.py',
            'lora': 'test_lora.py',
            'reward': 'test_reward_model.py',
            'generate': 'test_generate.py',
        }
        if args.module in module_map:
            cmd.append(str(test_dir / module_map[args.module]))
        else:
            print_error(f"未知模块: {args.module}")
            print_info(f"可用模块: {', '.join(module_map.keys())}")
            return 1
    else:
        cmd.append(str(test_dir))

    # 详细输出
    if args.verbose:
        cmd.append('-v')

    # 快速测试（跳过慢速测试）
    if args.fast:
        cmd.extend(['-m', 'not slow'])

    # 覆盖率
    if args.coverage:
        cmd.extend(['--cov=.', '--cov-report=html', '--cov-report=term-missing'])

    # 失败时停止
    if args.exitfirst:
        cmd.append('-x')

    # 显示测试时长
    if args.durations:
        cmd.extend(['--durations', str(args.durations)])

    # 并行运行
    if args.parallel:
        try:
            import pytest_xdist
            cmd.extend(['-n', 'auto'])
        except ImportError:
            print_info("提示: 安装 pytest-xdist 可以并行运行测试")

    # 运行测试
    print_info(f"运行命令: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    # 打印总结
    print()
    if result.returncode == 0:
        print_header("测试通过 ✓")
    else:
        print_header("测试失败 ✗")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="MyLLM 测试运行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_tests.py                    # 运行所有测试
  python run_tests.py -v                 # 详细输出
  python run_tests.py --module tokenizer # 只测试分词器
  python run_tests.py --fast             # 跳过慢速测试
  python run_tests.py --coverage         # 生成覆盖率报告
  python run_tests.py -x                 # 遇到失败立即停止

可用模块:
  tokenizer  - 分词器测试
  model      - 模型架构测试
  config     - 配置测试
  lora       - LoRA 测试
  reward     - 奖励模型测试
  generate   - 生成器测试
        """
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细输出'
    )

    parser.add_argument(
        '--module', '-m',
        type=str,
        help='只运行指定模块的测试 (tokenizer/model/config/lora/reward/generate)'
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        help='只运行快速测试（跳过标记为 slow 的测试）'
    )

    parser.add_argument(
        '--coverage',
        action='store_true',
        help='生成代码覆盖率报告'
    )

    parser.add_argument(
        '-x', '--exitfirst',
        action='store_true',
        help='遇到第一个失败立即停止'
    )

    parser.add_argument(
        '--durations',
        type=int,
        default=0,
        help='显示最慢的 N 个测试'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='并行运行测试（需要 pytest-xdist）'
    )

    args = parser.parse_args()

    return run_tests(args)


if __name__ == '__main__':
    sys.exit(main())
