"""
LoRA 微调训练脚本
======================

使用 LoRA（Low-Rank Adaptation）对预训练模型进行高效微调。
只训练少量参数（通常 < 1%），大幅降低训练成本。

注意：本脚本使用 data/lora_data.json 作为训练数据，与 train.py 使用的
data/sft_data.json 分离，避免相互干扰。

═══════════════════════════════════════════════════════════════════════════════
LoRA 微调 vs 全量微调
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                         全量微调 (Full Fine-tuning)                          │
│                                                                              │
│   预训练模型 (100% 参数)                                                     │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │  Embedding │ Transformer Blocks │ LM Head                       │       │
│   │     W1     │    W2, W3, W4...   │   W_out                       │       │
│   │   (训练)    │      (训练)        │  (训练)                       │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│   问题：                                                                     │
│   - 需要大量 GPU 显存（存储所有梯度和优化器状态）                             │
│   - 训练速度慢                                                               │
│   - 容易过拟合（特别是小数据集）                                             │
│   - 每个任务需要保存完整模型副本                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         LoRA 微调 (< 1% 参数)                                │
│                                                                              │
│   预训练模型 (冻结)           LoRA 适配器 (可训练)                            │
│   ┌──────────────────┐       ┌──────────────┐                               │
│   │   W (冻结)        │   +   │   B × A      │                               │
│   │  768 × 768       │       │ (768×8)×(8×768)│                               │
│   │  = 589,824 参数  │       │ = 12,288 参数 │                               │
│   └──────────────────┘       └──────────────┘                               │
│                                                                              │
│   优势：                                                                     │
│   - 只需训练 ~2% 的参数                                                      │
│   - 显存占用大幅降低                                                         │
│   - 可以为不同任务保存不同的 LoRA 权重                                       │
│   - 推理时可以合并回原模型，无额外开销                                       │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
训练流程
═══════════════════════════════════════════════════════════════════════════════

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  1. 加载     │ →   │  2. 应用     │ →   │  3. 准备     │ →   │  4. 训练     │
    │  基础模型    │     │   LoRA      │     │   数据      │     │   循环      │
    └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
          ↓                   ↓                   ↓                   ↓
    加载 SFT 后的       冻结原始权重          加载 SFT 数据        只更新 LoRA
    预训练模型          添加 A、B 矩阵       创建 DataLoader        参数

使用方法：
    # 基础用法
    python train_lora.py

    # 自定义参数
    python train_lora.py --base_model checkpoints/sft_final.pt --epochs 5 --lr 1e-4

    # 修改 LoRA 配置
    python train_lora.py --lora_r 16 --lora_alpha 32 --target_modules c_attn c_proj
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import GPT, GPTConfig, MyLLM, MyLLMConfig
from tokenizer import MyLLMTokenizer
from lora import LoRAConfig, LoRATrainer, apply_lora_to_model, save_lora


class SFTDataset(Dataset):
    """
    SFT 数据集，用于 LoRA 微调

    ═══════════════════════════════════════════════════════════════════════════
    数据处理流程
    ═══════════════════════════════════════════════════════════════════════════

    原始数据格式：
    -------------
    [
        {"user": "你好", "assistant": "你好！有什么可以帮助你的？"},
        {"user": "1+1等于几", "assistant": "1+1等于2"}
    ]

    处理后的格式：
    -------------
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  原始对话：                                                              │
    │    user: "你好"                                                          │
    │    assistant: "你好！有什么可以帮助你的？"                               │
    │                                                                          │
    │  转换为训练格式：                                                        │
    │    "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！..."  │
    │                                                                          │
    │  Token 化：                                                              │
    │    [1, 100, 2, 你, 好, 3, 2, 1, 101, 2, 你, 好, ！, ...]                │
    │                                                                          │
    │  自回归目标构造：                                                        │
    │    input_ids:  [1, 100, 2, 你, 好, 3, 2, 1, 101, 2, 你, 好, ！]          │
    │    target_ids: [100, 2, 你, 好, 3, 2, 1, 101, 2, 你, 好, ！, 3]          │
    │                ↑ 每个位置预测下一个 token                                 │
    └─────────────────────────────────────────────────────────────────────────┘

    与 train.py 中 SFTDataset 的区别：
    ---------------------------------
    - train.py: 只对 assistant 部分计算 loss（使用 loss_mask）
    - 这里: 对整个序列计算 loss（简化版本，适合 LoRA 快速微调）

    为什么 LoRA 可以用简化版本？
    --------------------------
    1. LoRA 只训练少量参数，不容易过拟合
    2. 预训练模型已经学会了对话格式，不需要严格区分
    3. 简化实现，减少代码复杂度
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: MyLLMTokenizer,
        max_length: int = 128
    ):
        """
        参数：
            data_path: 数据文件路径（JSON 格式）
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # ============================================================
        # Step 1: 检查数据文件是否存在
        # ============================================================
        if not os.path.exists(data_path):
            print(f"⚠️ 数据文件不存在: {data_path}")
            print(f"正在生成示例数据...")

            # 确保目录存在
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            # 生成示例数据
            self._create_sample_data(data_path)

        # ============================================================
        # Step 2: 加载数据
        # ============================================================
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"从 {data_path} 加载了 {len(data)} 条对话")

        # ============================================================
        # Step 3: 处理每条对话
        # ============================================================
        for item in tqdm(data, desc="处理数据"):
            user_text = item.get("user", "")
            assistant_text = item.get("assistant", "")

            # 构建对话格式（ChatML 格式）
            # <|im_start|>user\n{用户输入}<|im_end|>\n
            # <|im_start|>assistant\n{助手回复}<|im_end|>
            prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
            response = f"{assistant_text}<|im_end|>"

            # Token 化
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response)

            # 合并
            input_ids = prompt_ids + response_ids

            # 截断（保证不超过最大长度）
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            if len(input_ids) < 2:
                continue

            # 创建自回归目标
            # 语言模型的目标：给定前 t 个 token，预测第 t+1 个 token
            # 序列 [A, B, C, D] → input=[A,B,C], target=[B,C,D]
            self.samples.append({
                "input_ids": input_ids[:-1],   # 去掉最后一个 token
                "target_ids": input_ids[1:]    # 去掉第一个 token
            })

        print(f"处理完成，共 {len(self.samples)} 个训练样本")

    def _create_sample_data(self, data_path: str):
        """
        创建示例数据文件

        如果数据文件不存在，自动生成一份示例数据

        参数:
            data_path: 数据文件保存路径
        """
        # ============================================================
        # 示例数据：包含多种类型的对话
        # ============================================================
        sample_data = [
            # ==================== 问候类 ====================
            {"user": "你好", "assistant": "你好！很高兴见到你，有什么可以帮助你的吗？"},
            {"user": "早上好", "assistant": "早上好！祝你今天过得愉快！"},
            {"user": "晚上好", "assistant": "晚上好！今天过得怎么样？"},
            {"user": "谢谢", "assistant": "不客气！如果还有其他问题，随时告诉我。"},
            {"user": "再见", "assistant": "再见！期待下次与你交流。"},

            # ==================== 问答类 ====================
            {"user": "什么是人工智能", "assistant": "人工智能是指由计算机系统所表现出的智能行为。它包括机器学习、深度学习、自然语言处理等技术，让计算机能够模拟人类的认知功能，如学习、推理和自我修正。"},
            {"user": "什么是机器学习", "assistant": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而不是通过明确编程。常见的算法包括监督学习、无监督学习和强化学习。"},
            {"user": "什么是深度学习", "assistant": "深度学习是机器学习的一种方法，使用多层神经网络来学习数据的表示。它在图像识别、语音识别和自然语言处理等领域取得了突破性进展。"},
            {"user": "什么是神经网络", "assistant": "神经网络是一种受人脑神经元结构启发的计算模型。它由多层节点（神经元）组成，通过调整连接权重来学习从输入到输出的复杂映射关系。"},

            # ==================== 数学计算类 ====================
            {"user": "1+1等于几", "assistant": "1+1等于2。"},
            {"user": "2+3等于几", "assistant": "2+3等于5。"},
            {"user": "5乘以6等于多少", "assistant": "5乘以6等于30。"},
            {"user": "100除以4等于多少", "assistant": "100除以4等于25。"},
            {"user": "10的平方是多少", "assistant": "10的平方是100。"},
            {"user": "3的立方是多少", "assistant": "3的立方是27。"},

            # ==================== 代码相关类 ====================
            {"user": "Python中如何定义一个函数", "assistant": "在Python中，使用def关键字定义函数。例如：\ndef my_function():\n    print(\"Hello, World!\")\n\n调用时直接写：my_function()"},
            {"user": "什么是变量", "assistant": "变量是程序中用于存储数据的容器。你可以把它想象成一个标签，指向计算机内存中的某个位置，方便后续使用和修改存储的数据。"},
            {"user": "什么是循环", "assistant": "循环是一种重复执行某段代码的控制结构。Python中最常用的是for循环和while循环。for循环用于遍历序列，while循环在条件为真时重复执行。"},

            # ==================== 常识类 ====================
            {"user": "天空为什么是蓝色的", "assistant": "天空呈现蓝色是因为瑞利散射现象。太阳光穿过大气层时，短波长的蓝光比长波长的红光更容易被空气分子散射到各个方向，所以我们看到的天空是蓝色的。"},
            {"user": "为什么会有四季", "assistant": "地球四季的形成主要是由于地球的自转轴倾斜23.5度，绕太阳公转时，不同地区接收到的太阳辐射强度不同，从而产生了春夏秋冬的季节变化。"},
            {"user": "水在什么温度结冰", "assistant": "在标准大气压下，纯水在0摄氏度（32华氏度）时结冰。"},
            {"user": "地球有多大", "assistant": "地球的直径约为12,742公里，表面积约为5.1亿平方公里，赤道周长约为40,075公里。"},

            # ==================== 逻辑推理类 ====================
            {"user": "如果A比B大，B比C大，那么A和C谁大", "assistant": "根据传递性，如果A > B且B > C，那么A > C，所以A比C大。"},
            {"user": "所有的猫都是动物，汤姆是猫，那么汤姆是什么", "assistant": "根据前提，既然所有的猫都是动物，而汤姆是猫，那么汤姆是动物。"},
            {"user": "今天是星期三，三天后是星期几", "assistant": "如果今天是星期三，那么三天后是星期六。"},

            # ==================== 帮助/指导类 ====================
            {"user": "如何学习编程", "assistant": "学习编程的建议：\n1. 选择一门语言作为入门（如Python）\n2. 学习基础语法和数据结构\n3. 多动手实践，做小项目\n4. 阅读别人的代码\n5. 坚持不懈，持续练习"},
            {"user": "如何提高写作能力", "assistant": "提高写作能力的方法：\n1. 多阅读优秀作品\n2. 坚持每天写作练习\n3. 学习写作技巧和结构\n4. 请他人给反馈意见\n5. 反复修改和润色自己的文章"},
            {"user": "如何保持健康", "assistant": "保持健康的生活方式：\n1. 均衡饮食，多吃蔬果\n2. 适量运动，每周至少150分钟\n3. 充足睡眠，每天7-8小时\n4. 保持良好心态，减少压力\n5. 定期体检，预防疾病"},

            # ==================== 文化类 ====================
            {"user": "什么是春节", "assistant": "春节是中国最隆重的传统节日，也是农历新年。它历史悠久，起源于早期人类的原始信仰与自然崇拜，是集祈福攘灾、欢庆娱乐和饮食为一体的民俗大节。"},
            {"user": "什么是中秋节", "assistant": "中秋节是中国的传统节日，在农历八月十五庆祝。这一天月亮最圆最亮，象征团圆，人们会吃月饼、赏月、与家人团聚。"},

            # ==================== 科技产品类 ====================
            {"user": "什么是CPU", "assistant": "CPU是中央处理器的缩写，是计算机的核心部件，负责执行程序指令和处理数据。它被称为计算机的\"大脑\"，决定了计算机的基本性能。"},
            {"user": "什么是GPU", "assistant": "GPU是图形处理器的缩写，最初用于处理图形渲染。由于其并行计算能力强大，现在也广泛用于深度学习、科学计算和加密货币挖矿等领域。"},
            {"user": "什么是内存", "assistant": "内存是计算机的临时存储设备，用于存放正在运行的程序和数据。它的读写速度很快，但断电后数据会丢失。内存大小直接影响计算机的多任务处理能力。"},

            # ==================== 推荐类 ====================
            {"user": "推荐一本好书", "assistant": "如果你对人工智能感兴趣，我推荐《人工智能：一种现代方法》。如果是小说，《百年孤独》是经典之作。想学理财，《富爸爸穷爸爸》很受欢迎。"},
            {"user": "推荐一部电影", "assistant": "如果你喜欢科幻片，推荐《星际穿越》。喜欢喜剧的话，《功夫》很经典。文艺片爱好者可以看《肖申克的救赎》，它常年位居各大电影排行榜前列。"},

            # ==================== 解释概念类 ====================
            {"user": "什么是大数据", "assistant": "大数据是指无法用传统软件工具在一定时间内处理的海量数据集合。它具有5V特点：大量、高速、多样、价值密度低、真实性。大数据技术包括数据采集、存储、处理分析和可视化等。"},
            {"user": "什么是云计算", "assistant": "云计算是一种通过互联网按需提供计算资源的服务模式。用户可以像使用水电一样使用计算能力、存储和数据库，无需自己购买和维护硬件设备。常见的云服务有AWS、Azure和阿里云。"},
            {"user": "什么是区块链", "assistant": "区块链是一种分布式账本技术，通过密码学方法保证数据的不可篡改性和可追溯性。它最初用于比特币，现在应用在供应链、医疗、金融等多个领域。"},

            # ==================== 生活技巧类 ====================
            {"user": "如何高效记忆", "assistant": "高效记忆的技巧：\n1. 理解后再记忆，不要死记硬背\n2. 使用联想记忆法，将新旧知识联系起来\n3. 定期复习，利用艾宾浩斯遗忘曲线\n4. 多感官参与，边看边说边写\n5. 保证充足睡眠，睡眠有助于记忆巩固"},
            {"user": "如何管理时间", "assistant": "时间管理建议：\n1. 使用番茄工作法：25分钟专注工作+5分钟休息\n2. 四象限法则：区分紧急和重要的事情\n3. 消除干扰：工作时关闭不必要的通知\n4. 学会拒绝：不要接受所有任务\n5. 定期回顾：每周总结和计划"},

            # ==================== 情感支持类 ====================
            {"user": "我感到很焦虑", "assistant": "我理解你的感受。焦虑是很常见的情绪，以下是一些应对方法：\n1. 深呼吸练习，放松身心\n2. 分析焦虑的根源，具体是什么让你担心\n3. 制定行动计划，把担忧转化为行动\n4. 适度运动，释放压力\n5. 如果情况严重，建议寻求专业心理咨询"},
            {"user": "我感觉很孤独", "assistant": "孤独感是每个人都会经历的。你可以尝试：\n1. 主动联系朋友或家人\n2. 参加兴趣小组或社交活动\n3. 培养爱好，转移注意力\n4. 做志愿者，帮助他人\n5. 如果长期困扰，考虑寻求心理咨询师的帮助"},

            # ==================== 历史类 ====================
            {"user": "谁发明了电灯", "assistant": "电灯的发明通常归功于托马斯·爱迪生。他在1879年成功制造出实用的碳丝灯泡。不过，在他之前已有许多科学家进行过相关实验，如英国的约瑟夫·斯旺。"},
            {"user": "什么是文艺复兴", "assistant": "文艺复兴是14-17世纪欧洲的一场思想文化运动，起源于意大利佛罗伦萨，后扩展到全欧洲。它强调人文主义，重视人的价值和尊严，在艺术、科学、文学等领域都取得了巨大成就，为现代文明奠定了基础。"},
        ]

        # 保存到文件
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 已生成 {len(sample_data)} 条示例数据到: {data_path}")
        print(f"   数据包含：问候、问答、数学、代码、常识、逻辑推理、生活技巧等类型")
        print(f"\n你可以在该文件中添加或修改数据，然后重新运行训练。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["input_ids"], dtype=torch.long),
            torch.tensor(sample["target_ids"], dtype=torch.long)
        )


def collate_fn(batch):
    """
    动态填充 batch

    ═══════════════════════════════════════════════════════════════════════════
    Batch 填充示意图
    ═══════════════════════════════════════════════════════════════════════════

    问题：不同样本长度不同，无法直接堆叠成张量

    解决方案：动态填充到当前 batch 中最长序列的长度

    示例：
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Batch 中的样本（填充前）：                                              │
    │    样本1: [A, B, C, D, E]     长度 5                                    │
    │    样本2: [F, G, H]           长度 3                                    │
    │    样本3: [I, J, K, L]        长度 4                                    │
    │                                                                          │
    │  找到最长长度：max_len = 5                                               │
    │                                                                          │
    │  填充后（input_ids，用 0 填充）：                                        │
    │    样本1: [A, B, C, D, E]     无需填充                                   │
    │    样本2: [F, G, H, 0, 0]     填充 2 个 0                                │
    │    样本3: [I, J, K, L, 0]     填充 1 个 0                                │
    │                                                                          │
    │  填充后（target_ids，用 -100 填充）：                                    │
    │    样本1: [B, C, D, E, X]     X 是原始目标                               │
    │    样本2: [G, H, X, -100, -100]  -100 会被 loss 忽略                     │
    │    样本3: [J, K, L, X, -100]                                             │
    └─────────────────────────────────────────────────────────────────────────┘

    为什么 target 用 -100 填充？
    --------------------------
    PyTorch 的 CrossEntropyLoss 默认 ignore_index=-100
    这意味着 target=-100 的位置不会计算 loss
    """
    input_ids = [item[0] for item in batch]
    target_ids = [item[1] for item in batch]

    # 找到最长序列
    max_len = max(len(ids) for ids in input_ids)

    # 填充
    padded_inputs = []
    padded_targets = []

    for inp, tgt in zip(input_ids, target_ids):
        pad_len = max_len - len(inp)
        # input 用 0 填充（pad_token_id）
        padded_inputs.append(
            torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
        )
        # target 用 -100 填充（会被 CrossEntropyLoss 忽略）
        padded_targets.append(
            torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return torch.stack(padded_inputs), torch.stack(padded_targets)


def load_base_model(
    model_path: str,
    vocab_path: str,
    device: torch.device
):
    """
    加载基础模型

    自动检测模型类型（GPT 或 MyLLM）并正确加载

    参数:
        model_path: 模型检查点路径
        vocab_path: 词表路径（用于获取 vocab_size）
        device: 计算设备

    返回:
        加载好的模型
    """
    print(f"加载基础模型: {model_path}")

    # ============================================================
    # Step 1: 加载检查点
    # ============================================================
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # ============================================================
    # Step 2: 获取 state_dict
    # ============================================================
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # ============================================================
    # Step 3: 检测模型类型并推断配置
    # ============================================================
    # 检查键名格式来判断模型类型
    # MyLLM 使用 "transformer_blocks"，GPT 使用 "blocks"
    has_transformer_blocks = any('transformer_blocks.' in k for k in state_dict.keys())
    has_blocks = any('blocks.' in k for k in state_dict.keys())

    # 获取配置（如果存在）
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        print(f"从检查点读取配置: {config_dict}")
    else:
        print("检查点中没有配置信息，从权重推断...")
        config_dict = {}

        # 从权重推断 vocab_size（从 tok_emb.weight 的形状）
        for key in state_dict.keys():
            if 'tok_emb.weight' in key or 'token_embedding.weight' in key:
                vocab_size = state_dict[key].shape[0]
                emb_dim = state_dict[key].shape[1]
                config_dict["vocab_size"] = vocab_size
                config_dict["emb_dim"] = emb_dim
                print(f"从权重推断 vocab_size: {vocab_size}, emb_dim: {emb_dim}")
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

        # 从 pos_emb.weight 推断 context_size
        for key in state_dict.keys():
            if 'pos_emb.weight' in key or 'position_embedding.weight' in key:
                context_size = state_dict[key].shape[0]
                config_dict["context_size"] = context_size
                print(f"从权重推断 context_size: {context_size}")
                break

        # 设置默认值
        config_dict.setdefault("vocab_size", 6400)
        config_dict.setdefault("emb_dim", 256)
        config_dict.setdefault("num_heads", 4)
        config_dict.setdefault("context_size", 256)
        config_dict.setdefault("dropout", 0.1)

        print(f"推断配置: vocab_size={config_dict['vocab_size']}, layers={config_dict.get('num_layers', '?')}, heads={config_dict['num_heads']}, emb_dim={config_dict['emb_dim']}")

    # ============================================================
    # Step 4: 创建模型（根据类型选择类）
    # ============================================================
    if has_transformer_blocks:
        # 使用 MyLLM 模型
        config = MyLLMConfig(**config_dict)
        model = MyLLM(config)
        print("使用 MyLLM 模型类")
    else:
        # 使用 GPT 模型
        config = GPTConfig(**config_dict)
        model = GPT(config)
        print("使用 GPT 模型类")

    # ============================================================
    # Step 5: 加载权重
    # ============================================================
    model.load_state_dict(state_dict)
    model = model.to(device)

    print(f"模型参数量: {model.get_num_params():,}")

    return model


def train_lora(args):
    """
    LoRA 训练主函数
    """
    print("=" * 60)
    print("LoRA 微调训练")
    print("=" * 60)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载分词器
    print("\n" + "-" * 40)
    print("步骤 1: 加载分词器")
    print("-" * 40)
    tokenizer = MyLLMTokenizer(vocab_path=args.vocab_path)

    # 2. 加载基础模型
    print("\n" + "-" * 40)
    print("步骤 2: 加载基础模型")
    print("-" * 40)
    model = load_base_model(args.base_model, args.vocab_path, device)

    # 3. 配置 LoRA
    print("\n" + "-" * 40)
    print("步骤 3: 配置 LoRA")
    print("-" * 40)
    lora_config = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules
    )

    # 应用 LoRA
    model = apply_lora_to_model(model, lora_config, verbose=True)
    model = model.to(device)

    # 4. 准备数据
    print("\n" + "-" * 40)
    print("步骤 4: 准备数据")
    print("-" * 40)
    dataset = SFTDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 5. 设置优化器
    print("\n" + "-" * 40)
    print("步骤 5: 设置优化器")
    print("-" * 40)

    # 只优化 LoRA 参数
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in lora_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1
    )

    # ========================================================================
    # 6. 训练循环
    # ========================================================================
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │                      LoRA 训练循环详解                               │
    # │                                                                      │
    # │   每个 Epoch：                                                       │
    # │   ┌──────────────────────────────────────────────────────────────┐  │
    # │   │  for batch in dataloader:                                     │  │
    # │   │      1. 前向传播: logits, loss = model(input, target)         │  │
    # │   │         ↓                                                     │  │
    # │   │      2. 清零梯度: optimizer.zero_grad()                       │  │
    # │   │         ↓                                                     │  │
    # │   │      3. 反向传播: loss.backward()                             │  │
    # │   │         ↓   计算 LoRA 参数的梯度（原始权重冻结，无梯度）       │  │
    # │   │      4. 梯度裁剪: clip_grad_norm_(max=1.0)                    │  │
    # │   │         ↓   防止梯度爆炸                                      │  │
    # │   │      5. 更新参数: optimizer.step()                            │  │
    # │   │         ↓   只更新 LoRA 的 A、B 矩阵                          │  │
    # │   └──────────────────────────────────────────────────────────────┘  │
    # │   Epoch 结束后：scheduler.step() 更新学习率                         │
    # └─────────────────────────────────────────────────────────────────────┘
    print("\n" + "-" * 40)
    print("步骤 6: 开始训练")
    print("-" * 40)

    training_log = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            # ============================================================
            # Step 1: 前向传播
            # ============================================================
            # 注意：这里的 model 已经应用了 LoRA
            # 前向传播时会同时计算：
            #   - 原始权重的输出: Wx
            #   - LoRA 增量: BAx × scaling
            #   - 最终输出: Wx + BAx × scaling
            _, loss = model(input_ids, target_ids)

            # ============================================================
            # Step 2: 清零梯度
            # ============================================================
            optimizer.zero_grad()

            # ============================================================
            # Step 3: 反向传播
            # ============================================================
            # 关键：只有 LoRA 参数（A、B 矩阵）有梯度
            # 原始权重被冻结（requires_grad=False），不参与梯度计算
            loss.backward()

            # ============================================================
            # Step 4: 梯度裁剪
            # ============================================================
            # 防止梯度爆炸，保证训练稳定性
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)

            # ============================================================
            # Step 5: 更新参数
            # ============================================================
            # 只更新 lora_params（A、B 矩阵）
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # ============================================================
        # Epoch 结束：更新学习率
        # ============================================================
        # 使用 Cosine Annealing：学习率从初始值逐渐降低到 0.1 * lr
        scheduler.step()

        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  学习率: {current_lr:.6f}")

        # 记录日志
        training_log.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": current_lr,
            "timestamp": datetime.now().isoformat()
        })

        # 保存检查点
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
            save_lora(model, checkpoint_dir, lora_config)

    # 7. 保存最终模型
    print("\n" + "-" * 40)
    print("步骤 7: 保存模型")
    print("-" * 40)

    final_dir = os.path.join(args.output_dir, "final")
    save_lora(model, final_dir, lora_config)

    # 保存训练日志
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    print(f"训练日志已保存: {log_path}")

    print("\n" + "=" * 60)
    print("LoRA 训练完成!")
    print("=" * 60)

    return model


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调训练")

    # 模型和数据路径
    parser.add_argument(
        "--base_model", type=str,
        default="checkpoints/sft_final.pt",
        help="基础模型路径"
    )
    parser.add_argument(
        "--vocab_path", type=str,
        default="checkpoints/vocab.json",
        help="词表路径"
    )
    parser.add_argument(
        "--data_path", type=str,
        default="data/lora_data.json",
        help="训练数据路径"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="checkpoints/lora",
        help="输出目录"
    )

    # LoRA 参数
    parser.add_argument(
        "--lora_r", type=int, default=8,
        help="LoRA 秩"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16,
        help="LoRA 缩放因子"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--target_modules", type=str, nargs="+",
        default=["c_attn", "c_proj"],
        help="要应用 LoRA 的模块（MyLLM: c_attn, c_proj, linear1, linear2）"
    )

    # 训练参数
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="批次大小"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="权重衰减"
    )
    parser.add_argument(
        "--max_length", type=int, default=128,
        help="最大序列长度"
    )
    parser.add_argument(
        "--save_every", type=int, default=1,
        help="每多少轮保存一次"
    )

    args = parser.parse_args()

    # 打印配置
    print("=" * 60)
    print("LoRA 训练配置")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)

    train_lora(args)


if __name__ == "__main__":
    main()
