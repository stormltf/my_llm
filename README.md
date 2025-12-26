# MyLLM - 从零手搓大模型

一个完整的大语言模型学习项目，从零实现 GPT 风格的语言模型，包含完整的 5 阶段训练流程。

## 项目简介

本项目旨在帮助开发者理解大语言模型的底层原理，通过亲手实现每个组件，打破对大模型的神秘感。

### 核心特性

- **完整的 Transformer 架构**：从头实现注意力机制、前馈网络等核心组件
- **字符级分词器**：简单易懂的中文分词实现
- **5 阶段训练流程**：Pretrain → SFT → Reward Model → RLHF → RLVF
- **CPU 可训练**：~3.7M 参数的迷你模型，普通电脑即可运行

## 项目结构

```
my_llm/
├── data/                   # 数据目录
│   ├── sft_data.json       # SFT 对话数据 (94条)
│   ├── reward_data.json    # 奖励模型偏好数据 (40对)
│   ├── rlvf_data.json      # RLVF 可验证任务 (40条)
│   └── pretrain_data.txt   # 预训练文本
├── checkpoints/            # 模型检查点
│   ├── pretrain_final.pt   # 预训练模型
│   ├── sft_final.pt        # SFT 模型
│   ├── reward_model.pt     # 奖励模型
│   ├── rlhf_final.pt       # RLHF 模型
│   └── rlvf_final.pt       # RLVF 模型
├── tokenizer.py            # 字符级分词器
├── model.py                # Transformer 模型
├── config.py               # 配置文件
├── train.py                # 完整训练流程
├── reward_model.py         # 奖励模型
├── rlhf.py                 # RLHF (PPO) 训练器
├── rlvf.py                 # RLVF 训练器
├── inference.py            # 推理脚本
└── requirements.txt        # 依赖包
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 完整训练（5 阶段）

```bash
# 运行完整训练流程
python train.py

# 跳过特定阶段
python train.py --skip-pretrain --skip-sft    # 只运行 RLHF/RLVF
python train.py --skip-rlhf --skip-rlvf       # 只运行 Pretrain/SFT
```

---

## 5 阶段训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练流程图                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Stage 1: Pretrain          Stage 2: SFT                      │
│   ┌─────────────┐            ┌─────────────┐                   │
│   │ 语言模型    │  ────────► │ 有监督微调   │                   │
│   │ 预训练      │            │ (对话能力)   │                   │
│   └─────────────┘            └─────────────┘                   │
│         │                           │                          │
│         ▼                           ▼                          │
│   Stage 3: Reward Model      Stage 4: RLHF (PPO)              │
│   ┌─────────────┐            ┌─────────────┐                   │
│   │ 训练奖励    │  ────────► │ 人类反馈     │                   │
│   │ 模型        │            │ 强化学习     │                   │
│   └─────────────┘            └─────────────┘                   │
│                                     │                          │
│                                     ▼                          │
│                              Stage 5: RLVF                     │
│                              ┌─────────────┐                   │
│                              │ 可验证反馈   │                   │
│                              │ 强化学习     │                   │
│                              └─────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心模块详解

### 1. 分词器（Tokenizer）

**文件**：`tokenizer.py`

分词器将文本转换为模型可以处理的数字序列。本项目使用**字符级分词**，简单直观。

#### 工作原理

```
输入文本: "你好，世界"
    ↓
分词过程: ["你", "好", "，", "世", "界"]
    ↓
查词表:   你→15, 好→23, ，→5, 世→89, 界→102
    ↓
输出ID:   [15, 23, 5, 89, 102]
```

#### 特殊 Token

| Token | ID | 用途 |
|-------|-----|------|
| `<PAD>` | 0 | 填充，对齐序列长度 |
| `<UNK>` | 1 | 未知字符 |
| `<BOS>` | 2 | 序列开始 |
| `<EOS>` | 3 | 序列结束 |
| `<\|im_start\|>` | 4 | 对话角色开始 |
| `<\|im_end\|>` | 5 | 对话角色结束 |

#### 代码示例

```python
from tokenizer import MyLLMTokenizer

# 创建分词器
tokenizer = MyLLMTokenizer()

# 从文本构建词表
tokenizer.build_vocab("你好世界，我是AI助手。")

# 编码
ids = tokenizer.encode("你好")  # [15, 23]

# 解码
text = tokenizer.decode([15, 23])  # "你好"

# 保存/加载
tokenizer.save("vocab.json")
tokenizer.load("vocab.json")
```

#### 对话格式

```
<|im_start|>user
你好，请介绍一下自己。<|im_end|>
<|im_start|>assistant
你好！我是 MyLLM，一个小型语言模型。<|im_end|>
```

---

### 2. RLHF（基于人类反馈的强化学习）

**文件**：`rlhf.py`, `reward_model.py`

RLHF 让模型学习人类偏好，生成更符合人类期望的回答。

#### 核心思想

```
传统训练：模型学习"预测下一个词"
    ↓
RLHF：模型学习"生成人类喜欢的回答"
```

#### 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                      RLHF 训练流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 收集偏好数据                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Prompt: "什么是机器学习？"                           │   │
│  │                                                     │   │
│  │  Chosen (好回答):                                    │   │
│  │  "机器学习是AI的分支，让计算机从数据中学习规律..."      │   │
│  │                                                     │   │
│  │  Rejected (差回答):                                  │   │
│  │  "机器学习就是机器在学习，很简单的。"                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  Step 2: 训练奖励模型                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RewardModel(chosen) > RewardModel(rejected)        │   │
│  │                                                     │   │
│  │  损失函数: Bradley-Terry Loss                        │   │
│  │  L = -log(σ(r_chosen - r_rejected))                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│  Step 3: PPO 强化学习                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. 策略模型生成回答                                  │   │
│  │  2. 奖励模型打分                                      │   │
│  │  3. 计算优势函数 (Advantage)                          │   │
│  │  4. PPO 裁剪更新策略                                  │   │
│  │                                                     │   │
│  │  L_PPO = min(r*A, clip(r, 1-ε, 1+ε)*A)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `clip_ratio` | PPO 裁剪系数，防止策略更新过大 | 0.2 |
| `kl_coef` | KL 散度惩罚，保持与原策略接近 | 0.01 |
| `num_episodes` | 训练轮数 | 50 |

#### 代码示例

```python
from rlhf import PPOTrainer, RLHFConfig
from reward_model import RewardModel, RewardModelTrainer

# 1. 训练奖励模型
reward_model = RewardModel.from_pretrained(base_model, config)
rm_trainer = RewardModelTrainer(reward_model, tokenizer, config)
rm_trainer.train(reward_data, epochs=3)

# 2. PPO 训练
rlhf_config = RLHFConfig(
    clip_ratio=0.2,
    kl_coef=0.01,
    num_episodes=50
)
ppo_trainer = PPOTrainer(policy_model, reward_model, tokenizer, config, rlhf_config)
ppo_trainer.train(prompts)
```

---

### 3. RLVF（基于可验证反馈的强化学习）

**文件**：`rlvf.py`

RLVF 使用**可验证的正确答案**作为奖励信号，无需人类标注。

#### 与 RLHF 的区别

| 特性 | RLHF | RLVF |
|------|------|------|
| 奖励来源 | 人类偏好标注 | 自动验证器 |
| 适用场景 | 开放式问题 | 有明确答案的问题 |
| 标注成本 | 高 | 低 |
| 奖励准确性 | 主观 | 客观 |

#### 支持的验证类型

**1. 数学验证器 (MathVerifier)**

```python
# 验证数学答案
prompt: "3 + 5 等于多少？"
expected: "8"
response: "答案是8。"  → reward = 1.0 (正确)
response: "答案是7。"  → reward = -0.5 (错误)
```

**2. 逻辑验证器 (LogicVerifier)**

```python
# 验证逻辑推理
prompt: "所有猫都是动物，小花是猫，请问小花是动物吗？"
expected: "是"
keywords: ["是", "正确", "对"]
response: "是的，小花是动物。"  → reward = 1.0
```

#### 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                      RLVF 训练流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │  给定任务    │ ──► │  模型生成   │ ──► │  验证器判断  │   │
│  │  (数学/逻辑) │     │  回答       │     │  对/错      │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
│                                                 │           │
│                                                 ▼           │
│                                          ┌─────────────┐   │
│                                          │  计算奖励    │   │
│                                          │  正确: +1.0  │   │
│                                          │  错误: -0.5  │   │
│                                          └─────────────┘   │
│                                                 │           │
│                                                 ▼           │
│                                          ┌─────────────┐   │
│                                          │  策略梯度    │   │
│                                          │  更新模型    │   │
│                                          └─────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 代码示例

```python
from rlvf import RLVFTrainer, RLVFConfig, MathVerifier, LogicVerifier

# 配置
rlvf_config = RLVFConfig(
    correct_reward=1.0,      # 正确答案奖励
    incorrect_reward=-0.5,   # 错误答案惩罚
    num_iterations=30,       # 训练迭代次数
    samples_per_task=2       # 每个任务采样次数
)

# 训练
trainer = RLVFTrainer(model, tokenizer, config, rlvf_config)
history = trainer.train(tasks, batch_size=4)

# 评估
results = trainer.evaluate(test_tasks)
print(f"准确率: {results['accuracy']:.2%}")
```

#### 任务数据格式

```json
{
    "type": "math",
    "prompt": "计算 15 + 27 = ?",
    "expected_answer": "42"
}
```

```json
{
    "type": "logic",
    "prompt": "如果今天是周一，明天是周几？",
    "expected_answer": "周二",
    "keywords": ["周二", "星期二", "二"]
}
```

---

## 核心概念详解

### Transformer 架构

```
输入 "你好" [token_ids]
      ↓
┌─────────────────────────────────────┐
│  Token Embedding + Position Embedding │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Transformer Block × N              │
│  ├── LayerNorm                      │
│  ├── Self-Attention (词间交流)       │
│  ├── Residual Connection            │
│  ├── LayerNorm                      │
│  ├── Feed-Forward (深度处理)         │
│  └── Residual Connection            │
└─────────────────────────────────────┘
      ↓
┌─────────────────────────────────────┐
│  Output Projection → Vocabulary     │
└─────────────────────────────────────┘
      ↓
预测下一个 token 的概率分布
```

### Self-Attention

```
Attention(Q, K, V) = softmax(QK^T / √d) × V

Q (Query):  "我想找什么信息"
K (Key):    "我有什么标签"
V (Value):  "我的实际内容"
```

---

## 训练数据

### SFT 数据 (94条)

覆盖多种对话场景：
- 自我介绍、AI/ML 知识
- 代码解释、数学逻辑
- 创意写作、安全边界

### 奖励数据 (40对)

每条包含：
- `prompt`: 用户问题
- `chosen`: 好的回答
- `rejected`: 差的回答

### RLVF 数据 (40条)

- 数学题：加减乘除、应用题
- 逻辑题：推理、判断

---

## 模型配置

```python
from config import get_mini_config

config = get_mini_config()
# vocab_size: 6400
# emb_dim: 256
# num_heads: 4
# num_layers: 4
# context_size: 256
# 参数量: ~3.7M
```

---

## 参考资料

- 《Build a Large Language Model (From Scratch)》
- "Attention Is All You Need" (Transformer 原论文)
- "Training language models to follow instructions with human feedback" (InstructGPT/RLHF)
- "Constitutional AI" (Anthropic)

---

## 许可证

MIT License

---

**祝学习愉快！**
