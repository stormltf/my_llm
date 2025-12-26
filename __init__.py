"""
MyLLM - 从零实现大语言模型
=================================

一个完整的、可学习的小型中文语言模型实现。
约 5M 参数，支持 CPU/MPS/CUDA 训练。

核心模块：
---------
- config: 模型配置 (MyLLMConfig)
- tokenizer: 分词器 (MyLLMTokenizer)
- model: Transformer 模型 (MyLLM)
- train: 预训练 + SFT 训练脚本
- inference: 推理和对话 (MyLLMChat)

快速开始：
---------
>>> from my_llm import MyLLMConfig, MyLLM, MyLLMTokenizer
>>>
>>> # 创建配置
>>> config = MyLLMConfig()
>>>
>>> # 创建分词器
>>> tokenizer = MyLLMTokenizer()
>>> tokenizer.build_vocab("你好世界")
>>>
>>> # 创建模型
>>> model = MyLLM(config)
>>>
>>> # 编码输入
>>> input_ids = tokenizer.encode("你好", return_tensors="pt")
>>>
>>> # 生成
>>> output_ids = model.generate(input_ids, max_new_tokens=10)
>>> print(tokenizer.decode(output_ids[0]))

使用训练好的模型：
-----------------
>>> from inference import MyLLMChat
>>> chat = MyLLMChat("checkpoints/sft_final.pt", "checkpoints/vocab.json")
>>> response = chat.chat("你好")
>>> print(response)
"""

from .config import MyLLMConfig, get_mini_config, get_small_config
from .tokenizer import MyLLMTokenizer, create_chinese_vocab
from .model import MyLLM

__version__ = "1.0.0"
__author__ = "MyLLM Team"

__all__ = [
    "MyLLMConfig",
    "get_mini_config",
    "get_small_config",
    "MyLLMTokenizer",
    "create_chinese_vocab",
    "MyLLM",
]
