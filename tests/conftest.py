"""
PyTest 配置和共享 fixtures
"""

import sys
import os
import pytest
import torch

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MyLLMConfig, get_mini_config
from model import GPT, GPTConfig
from tokenizer import BPETokenizer, MyLLMTokenizer


@pytest.fixture
def device():
    """获取测试设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mini_config():
    """迷你版模型配置 (用于快速测试)"""
    return MyLLMConfig(
        model_name="test-mini",
        vocab_size=100,
        emb_dim=32,
        num_heads=2,
        num_layers=2,
        context_size=64,
        dropout=0.0,  # 测试时关闭 dropout
    )


@pytest.fixture
def gpt_config(mini_config):
    """GPT 配置 (基于 mini_config)"""
    return GPTConfig(
        vocab_size=mini_config.vocab_size,
        emb_dim=mini_config.emb_dim,
        num_heads=mini_config.num_heads,
        num_layers=mini_config.num_layers,
        context_size=mini_config.context_size,
        dropout=mini_config.dropout,
    )


@pytest.fixture
def small_model(gpt_config, device):
    """创建一个小型测试模型"""
    model = GPT(gpt_config)
    model = model.to(device)
    model.eval()
    return model


@pytest.fixture
def sample_texts():
    """示例训练文本"""
    return [
        "我 喜欢 学习 人工智能",
        "人工智能 是 未来 的 趋势",
        "深度 学习 是 机器 学习 的 分支",
        "自然 语言 处理 很 有趣",
        "大模型 改变 了 世界",
    ]


@pytest.fixture
def trained_tokenizer(sample_texts):
    """训练好的分词器"""
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.fit(sample_texts, verbose=False)
    return tokenizer


@pytest.fixture
def myllm_tokenizer():
    """MyLLM 分词器 (未训练)"""
    return MyLLMTokenizer(vocab_size=100)


# 添加 pytest 选项
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--rungpu", action="store_true", default=False, help="run GPU tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--rungpu"):
        skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
