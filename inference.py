"""
MyLLM 推理脚本
======================
这个文件实现了模型的推理和对话功能。

推理原理：
--------
1. 将用户输入转换为 Token IDs
2. 逐个预测下一个 Token（自回归生成）
3. 应用采样策略（Temperature、Top-k、Top-p）
4. 将生成的 Token IDs 转换回文本

采样策略：
--------
1. Temperature: 控制生成的随机性
   - 低温(<1): 更确定，选概率最高的词
   - 高温(>1): 更随机，给低概率词更多机会

2. Top-k: 只从概率最高的 k 个词中采样
3. Top-p: 只从累积概率达到 p 的词中采样

作者：MyLLM Team
"""

import os
import torch
from typing import Optional

from config import MyLLMConfig
from tokenizer import MyLLMTokenizer
from model import MyLLM


class MyLLMChat:
    """
    MyLLM 对话系统

    提供简单的对话接口，支持：
    1. 文本续写（预训练模式）
    2. 对话问答（SFT 模式）
    """

    def __init__(
        self,
        model_path: str = "checkpoints/sft_final.pt",
        vocab_path: str = "checkpoints/vocab.json",
        device: Optional[str] = None
    ):
        """
        参数：
            model_path: 模型检查点路径
            vocab_path: 词表路径
            device: 运行设备 (cuda/cpu)
        """
        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        # 加载分词器
        print("加载分词器...")
        self.tokenizer = MyLLMTokenizer(vocab_path)

        # 加载模型
        print("加载模型...")
        self._load_model(model_path)

        print("模型加载完成！")

    def _load_model(self, model_path: str):
        """
        加载模型

        安全性说明：
        - 使用 weights_only=True 安全加载模型权重
        - 从单独的 JSON 文件加载配置，避免 pickle 安全风险
        - 同时支持新格式（分离的配置文件）和旧格式（兼容性）
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 构建配置文件路径
        # 例如: sft_final.pt -> sft_final_config.json
        base_path = model_path.rsplit('.', 1)[0]
        config_path = f"{base_path}_config.json"

        # 尝试从 JSON 加载配置（新格式，安全）
        if os.path.exists(config_path):
            self.config = MyLLMConfig.load(config_path)
            # 安全加载模型权重
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        else:
            # 兼容旧格式：从 checkpoint 加载（不安全，但保持向后兼容）
            # 显示警告信息
            print("警告: 使用旧格式加载模型，建议重新训练以使用更安全的存储格式")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.config = checkpoint["config"]
            state_dict = checkpoint["model_state_dict"]

        # 创建模型
        self.model = MyLLM(self.config)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.2,
        show_input: bool = True
    ) -> str:
        """
        文本生成

        参数：
            prompt: 输入提示词
            max_new_tokens: 最多生成多少个新 token
            temperature: 温度参数
            top_k: Top-k 采样参数
            top_p: Top-p 采样参数（可选）
            repetition_penalty: 重复惩罚系数
            show_input: 是否在输出中包含输入

        返回：
            生成的文本
        """
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        input_len = input_ids.shape[1]  # 记录输入长度

        # 生成
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty
            )

        if not show_input:
            # 只解码新生成的部分
            new_ids = output_ids[0][input_len:]
            generated_text = self.tokenizer.decode(new_ids)
        else:
            # 解码全部
            generated_text = self.tokenizer.decode(output_ids[0])

        # 移除重复内容
        generated_text = self._remove_repetition(generated_text)

        return generated_text

    def _remove_repetition(self, text: str) -> str:
        """
        移除重复内容和杂乱内容

        检测并移除文本中的重复部分，保留第一次出现的完整内容
        使用多种策略：
        1. 检测杂乱内容标记
        2. 检测字符级重复（如 aaa, assistant重复变体）
        3. N-gram 重复检测
        4. 句子级重复检测
        """
        import re

        if len(text) < 10:
            return text

        # 策略0：检测杂乱内容的标记词
        # 这些通常是模型在生成完正确答案后混入的无关内容
        noise_markers = [
            "我的均值", "然后标准化", "我的。建议", "希望我的反馈",
            "我是常用的", "4. 。", "我的加速", "我的编程助手。我的",
            "我的。", "基础知识。我的", "asistant", "aassistant",
            "ssistant", "assistantant"
        ]
        # 找到所有 marker 中位置最早的一个
        earliest_pos = len(text)
        for marker in noise_markers:
            if marker in text.lower():
                pos = text.lower().find(marker)
                if pos < earliest_pos:
                    earliest_pos = pos

        # 如果找到了 marker，在其之前截断
        if earliest_pos < len(text):
            last_period = text[:earliest_pos].rfind('。')
            last_exclaim = text[:earliest_pos].rfind('！')
            cut_pos = max(last_period, last_exclaim)
            if cut_pos > 10:
                return text[:cut_pos + 1]

        # 策略0.5：检测字符级重复（如连续相同字符 aaa, 或者碎片如 ali）
        # 检测连续相同字符超过3个
        char_repeat = re.search(r'(.)\1{3,}', text)
        if char_repeat:
            cut_pos = char_repeat.start()
            last_period = text[:cut_pos].rfind('。')
            if last_period > 10:
                return text[:last_period + 1]

        # 检测短碎片模式（2-3个无意义字符的重复）
        fragment_pattern = re.search(r'([a-zA-Z]{1,3})\1{2,}', text)
        if fragment_pattern:
            cut_pos = fragment_pattern.start()
            last_period = text[:cut_pos].rfind('。')
            if last_period > 10:
                return text[:last_period + 1]

        # 策略1：检测 N-gram 重复（滑动窗口）
        # 如果一个短语（4-8个字）重复出现，在第二次出现时截断
        for n in [8, 6, 4]:  # 从长到短检测
            for i in range(len(text) - n * 2):
                pattern = text[i:i + n]
                # 查找这个模式在后面是否重复出现
                rest = text[i + n:]
                if pattern in rest:
                    # 找到重复的位置
                    repeat_pos = rest.find(pattern)
                    # 在重复开始前截断
                    cut_pos = i + n + repeat_pos
                    # 尝试在句号处截断
                    last_period = text[:cut_pos].rfind('。')
                    if last_period > len(text) // 3:  # 确保不会截断太多
                        return text[:last_period + 1]
                    # 否则直接在重复处截断
                    return text[:cut_pos].rstrip('，。！？!?,.')

        # 策略2：句子级重复检测
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in "。！？!?.":
                sentences.append(current)
                current = ""
        if current:
            sentences.append(current)

        # 保留不重复的句子
        seen = set()
        result = []
        for s in sentences:
            s_clean = s.strip()
            if s_clean and s_clean not in seen:
                seen.add(s_clean)
                result.append(s)
            else:
                break  # 遇到重复就停止

        return "".join(result) if result else text

    def chat(
        self,
        user_input: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50
    ) -> str:
        """
        对话模式

        使用 SFT 格式进行对话

        参数：
            user_input: 用户输入
            max_new_tokens: 最多生成多少个新 token
            temperature: 温度参数
            top_k: Top-k 采样参数

        返回：
            模型的回答
        """
        # 构造 SFT 格式的输入
        prompt = (
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 生成
        response = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            show_input=False
        )

        # 提取 assistant 回答部分
        # 去掉可能的结束标记
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        # 移除重复内容
        response = self._remove_repetition(response)

        return response.strip()

    def interactive_chat(self):
        """
        交互式对话

        在命令行中与模型对话
        """
        print("\n" + "=" * 60)
        print("MyLLM 交互式对话")
        print("=" * 60)
        print("提示：")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("  - 输入 'clear' 清屏")
        print("  - 输入 'mode:pretrain' 切换到续写模式")
        print("  - 输入 'mode:chat' 切换到对话模式")
        print("=" * 60)

        mode = "chat"  # 默认对话模式
        print(f"当前模式: {mode}")

        while True:
            try:
                # 获取用户输入
                user_input = input("\n你: ").strip()

                if not user_input:
                    continue

                # 检查特殊命令
                if user_input.lower() in ["quit", "exit"]:
                    print("再见！")
                    break

                if user_input.lower() == "clear":
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue

                if user_input.lower() == "mode:pretrain":
                    mode = "pretrain"
                    print(f"已切换到: {mode} 模式（文本续写）")
                    continue

                if user_input.lower() == "mode:chat":
                    mode = "chat"
                    print(f"已切换到: {mode} 模式（对话问答）")
                    continue

                # 生成回答
                if mode == "chat":
                    response = self.chat(user_input)
                else:
                    response = self.generate(user_input, show_input=False)

                print(f"\nMyLLM: {response}")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}")


def demo_generation():
    """演示文本生成"""
    print("=" * 60)
    print("MyLLM 文本生成演示")
    print("=" * 60)

    # 检查模型文件是否存在
    model_path = "checkpoints/sft_final.pt"
    vocab_path = "checkpoints/vocab.json"

    if not os.path.exists(model_path):
        print(f"\n模型文件不存在: {model_path}")
        print("请先运行 train.py 进行训练！")
        print("\n演示使用随机初始化的模型...")

        # 创建临时模型进行演示
        from config import get_mini_config
        from tokenizer import create_chinese_vocab

        config = get_mini_config()
        tokenizer = create_chinese_vocab()
        config.vocab_size = tokenizer.vocab_size
        model = MyLLM(config)

        # 演示生成
        print("\n[预训练模式] 输入: '人工智能是'")
        input_ids = tokenizer.encode("人工智能是", return_tensors="pt")
        output_ids = model.generate(input_ids, max_new_tokens=30, temperature=1.0)
        print(f"输出: {tokenizer.decode(output_ids[0])}")
        print("\n(由于模型未训练，输出是随机的)")
        return

    # 加载训练好的模型
    chat = MyLLMChat(model_path, vocab_path)

    # 演示预训练模式（文本续写）
    print("\n" + "-" * 40)
    print("[预训练模式] 文本续写演示")
    print("-" * 40)

    prompts = [
        "人工智能是",
        "今天天气",
        "学习编程",
    ]

    for prompt in prompts:
        print(f"\n输入: '{prompt}'")
        response = chat.generate(prompt, max_new_tokens=50)
        print(f"输出: {response}")

    # 演示对话模式
    print("\n" + "-" * 40)
    print("[对话模式] 问答演示")
    print("-" * 40)

    questions = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "如何学习编程？",
    ]

    for question in questions:
        print(f"\n用户: {question}")
        response = chat.chat(question, max_new_tokens=80)
        print(f"MyLLM: {response}")


def main():
    """主函数"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # 演示模式
        demo_generation()
    else:
        # 交互模式
        model_path = "checkpoints/sft_final.pt"
        vocab_path = "checkpoints/vocab.json"

        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            print("请先运行 train.py 进行训练！")
            print("\n可以运行 'python inference.py demo' 查看演示")
            return

        chat = MyLLMChat(model_path, vocab_path)
        chat.interactive_chat()


if __name__ == "__main__":
    main()
