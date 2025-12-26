"""
LoRA 模型推理脚本
======================

使用 LoRA 微调后的模型进行推理
"""

import os
import torch
from typing import Optional

from model import GPT, GPTConfig
from tokenizer import MyLLMTokenizer
from lora import LoRAConfig, load_lora, merge_lora


class LoRAInference:
    """LoRA 模型推理"""

    def __init__(
        self,
        base_model_path: str = "checkpoints/sft_final.pt",
        lora_path: str = "checkpoints/lora/final",
        vocab_path: str = "checkpoints/vocab.json",
        device: Optional[str] = None,
        merge: bool = False
    ):
        """
        参数：
            base_model_path: 基础模型路径
            lora_path: LoRA 权重路径
            vocab_path: 词表路径
            device: 运行设备
            merge: 是否合并 LoRA 权重（合并后推理更快，但无法切换 LoRA）
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"使用设备: {self.device}")

        # 加载分词器
        print("加载分词器...")
        self.tokenizer = MyLLMTokenizer(vocab_path)

        # 加载基础模型
        print("加载基础模型...")
        self.model = self._load_base_model(base_model_path)

        # 加载 LoRA
        print(f"加载 LoRA 权重: {lora_path}")
        self.model = load_lora(self.model, lora_path, apply_if_needed=True)
        self.model.to(self.device)
        self.model.eval()

        # 是否合并权重
        if merge:
            print("合并 LoRA 权重到模型...")
            merge_lora(self.model)

        print("模型加载完成！")

    def _load_base_model(self, model_path: str):
        """加载基础模型"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # 获取 state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # 从权重推断配置
        config_dict = {}

        # 从 tok_emb.weight 推断 vocab_size 和 emb_dim
        for key in state_dict.keys():
            if 'tok_emb.weight' in key or 'token_embedding.weight' in key:
                config_dict["vocab_size"] = state_dict[key].shape[0]
                config_dict["emb_dim"] = state_dict[key].shape[1]
                break

        # 从 pos_emb.weight 推断 context_size
        for key in state_dict.keys():
            if 'pos_emb.weight' in key or 'position_embedding.weight' in key:
                config_dict["context_size"] = state_dict[key].shape[0]
                break

        # 从权重推断层数
        has_blocks = any('blocks.' in k for k in state_dict.keys())
        if has_blocks:
            num_layers = max(
                int(k.split('.')[1])
                for k in state_dict.keys()
                if 'blocks.' in k and '.ln_1' in k
            ) + 1
            config_dict["num_layers"] = num_layers

        # 设置默认值
        config_dict.setdefault("vocab_size", 6400)
        config_dict.setdefault("emb_dim", 256)
        config_dict.setdefault("num_heads", 4)
        config_dict.setdefault("context_size", 256)
        config_dict.setdefault("dropout", 0.1)

        # 创建模型
        config = GPTConfig(**config_dict)
        model = GPT(config)
        model.load_state_dict(state_dict)

        return model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.2,
    ) -> str:
        """生成文本"""
        # 对话格式
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # 编码
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        input_len = input_ids.shape[1]

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

        # 解码（只取新生成的部分）
        new_ids = output_ids[0][input_len:]
        response = self.tokenizer.decode(new_ids)

        # 去掉结束标记
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]

        return response.strip()

    def interactive_chat(self):
        """交互式对话"""
        print("\n" + "=" * 60)
        print("LoRA 模型交互式对话")
        print("=" * 60)
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n你: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("再见！")
                    break

                response = self.generate(user_input)
                print(f"\n助手: {response}")

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}")


def main():
    """主函数"""
    import sys

    # 解析参数
    base_model = "checkpoints/sft_final.pt"
    lora_path = "checkpoints/lora/final"
    vocab_path = "checkpoints/vocab.json"

    if len(sys.argv) > 1:
        lora_path = sys.argv[1]

    print("=" * 60)
    print("LoRA 推理")
    print("=" * 60)
    print(f"基础模型: {base_model}")
    print(f"LoRA 权重: {lora_path}")
    print(f"词表: {vocab_path}")
    print("=" * 60)

    # 创建推理对象
    inference = LoRAInference(
        base_model_path=base_model,
        lora_path=lora_path,
        vocab_path=vocab_path,
        merge=False  # 设为 True 可合并权重，推理更快
    )

    # 交互式对话
    inference.interactive_chat()


if __name__ == "__main__":
    main()
