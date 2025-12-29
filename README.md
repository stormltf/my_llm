# MyLLM - 从零手搓大模型

一个完整的大语言模型学习项目，从零实现 GPT 风格的语言模型，包含完整的 5 阶段训练流程。

## 项目简介

本项目旨在帮助开发者理解大语言模型的底层原理，通过亲手实现每个组件，打破对大模型的神秘感。

### 核心特性

- **完整的 Transformer 架构**：从头实现注意力机制、前馈网络等核心组件
- **BPE 分词器**：基于 Byte Pair Encoding 的高效子词分词
- **5 阶段训练流程**：Pretrain → SFT → Reward Model → RLHF → RLVF
- **CPU 可训练**：~5M 参数的迷你模型，普通电脑即可运行
- **LoRA 高效微调**：支持低秩适应，参数量仅需 1-2%

---

## 📖 学习路径引导 (For 新人)

作为大语言模型的学习项目，建议按照以下顺序学习，由浅入深逐步掌握：

### 阶段 1：基础准备 (1-2 天)
- **学习目标**：理解 LLM 的基本概念和项目结构
- **阅读文件**：
  1. `README.md` - 完整阅读，了解项目全貌
  2. `tokenizer.py` - 学习 BPE 分词原理
  3. `config.py` - 了解模型和训练的配置参数
- **实验**：
  ```bash
  # 运行分词器测试，观察结果
  python -m pytest tests/test_tokenizer.py -v
  ```

### 阶段 2：模型架构 (2-3 天)
- **学习目标**：掌握 Transformer 架构的核心原理
- **阅读文件**：
  1. `model.py` - 从 LayerNorm → FeedForward → CausalSelfAttention → TransformerBlock → GPT 顺序阅读
  2. 重点理解 `CausalSelfAttention` 类的实现
- **实验**：
  ```bash
  # 运行模型架构测试
  python -m pytest tests/test_model.py -v
  ```

### 阶段 3：训练流程 (3-5 天)
- **学习目标**：理解大语言模型的训练流程
- **阅读文件**：
  1. `train.py` - 重点阅读 Pretrain 和 SFT 部分
- **实验**：
  ```bash
  # 运行完整训练流程（约 30 分钟）
  python train.py
  ```

### 阶段 4：强化学习阶段 (2-3 天)
- **学习目标**：理解 RLHF/RLVF 的工作原理
- **阅读文件**：
  1. `reward_model.py` - 奖励模型实现
  2. `rlhf.py` - PPO 算法实现
  3. `rlvf.py` - 可验证反馈实现
- **实验**：
  ```bash
  # 只运行强化学习阶段的训练
  python train.py --skip-pretrain --skip-sft --skip-reward
  ```

### 阶段 5：高级功能 (2-3 天)
- **学习目标**：掌握 LoRA 高效微调等高级功能
- **阅读文件**：
  1. `lora.py` - LoRA 实现原理
  2. `train_lora.py` - LoRA 训练脚本
- **实验**：
  ```bash
  # 运行 LoRA 微调
  python train_lora.py
  ```

---

## 快速开始

### 环境要求

- **Python**: 3.8+
- **操作系统**: Linux / macOS / Windows
- **硬件**: CPU 即可运行（GPU 可加速）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-repo/my_llm.git
cd my_llm

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
```

### 5分钟上手

```bash
# 运行完整训练流程（约10-30分钟，取决于CPU）
python3 train.py

# 训练完成后，启动交互式对话
python3 generate.py --interactive
```

---

## 整体架构

### 训练流程全景图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           大模型训练完整流程                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        第一部分：数据准备                             │   │
│  │  ┌───────────┐                                                      │   │
│  │  │ 原始文本   │ ──► Tokenizer ──► Token IDs ──► 训练数据            │   │
│  │  └───────────┘     (分词器)       [15,23,5...]                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        第二部分：模型架构                             │   │
│  │                                                                     │   │
│  │  Token IDs ──► Embedding ──► Transformer Blocks ──► Output Layer   │   │
│  │                                    ↓                                │   │
│  │                         [Self-Attention + FFN] × N                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        第三部分：训练流程                             │   │
│  │                                                                     │   │
│  │   Stage 1        Stage 2        Stage 3        Stage 4    Stage 5  │   │
│  │  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐  ┌────────┐ │   │
│  │  │Pretrain│ ─► │  SFT   │ ─► │ Reward │ ─► │  RLHF  │─►│  RLVF  │ │   │
│  │  │预训练  │    │监督微调│    │  Model │    │  PPO   │  │可验证RL│ │   │
│  │  └────────┘    └────────┘    └────────┘    └────────┘  └────────┘ │   │
│  │      │              │             │             │           │      │   │
│  │      ▼              ▼             ▼             ▼           ▼      │   │
│  │   语言建模      对话能力       偏好学习      策略优化     精确推理  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 各阶段目标

| 阶段 | 名称 | 目标 | 数据类型 |
|------|------|------|----------|
| 0 | Tokenizer | 构建 BPE 词表，文本↔数字转换 | 大量无标注文本 |
| 1 | Pretrain | 学习语言规律，预测下一个词 | 大量无标注文本 |
| 2 | SFT | 学习对话格式，获得指令遵循能力 | 人工标注的对话数据 |
| 3 | Reward Model | 学习人类偏好，区分好坏回答 | 偏好对比数据 |
| 4 | RLHF | 生成符合人类偏好的回答 | 奖励模型 + PPO |
| 5 | RLVF | 提升精确推理能力 | 可验证的任务 |

### 项目结构

```
my_llm/
├── tokenizer.py            # [阶段0] BPE 分词器
├── model.py                # [模型] Transformer 架构
├── config.py               # [配置] 模型和训练参数
├── train.py                # [训练] 完整 5 阶段流程
├── reward_model.py         # [阶段3] 奖励模型
├── rlhf.py                 # [阶段4] RLHF (PPO) 训练器
├── rlvf.py                 # [阶段5] RLVF 训练器
├── generate.py             # [生成] 文本生成与对话
├── inference.py            # [推理] 推理脚本
├── lora.py                 # [高效微调] LoRA 实现
├── train_lora.py           # [高效微调] LoRA 训练脚本
├── inference_lora.py       # [高效微调] LoRA 推理
├── data/
│   ├── pretrain_data.txt   # 预训练文本
│   ├── sft_data.json       # SFT 对话数据 (94条)
│   ├── reward_data.json    # 奖励数据 (40对)
│   └── rlvf_data.json      # RLVF 任务 (40条)
├── tests/                  # 单元测试
│   ├── test_tokenizer.py   # 分词器测试
│   ├── test_model.py       # 模型测试
│   ├── test_config.py      # 配置测试
│   ├── test_lora.py        # LoRA 测试
│   ├── test_generate.py    # 生成器测试
│   ├── test_reward_model.py # 奖励模型测试
│   ├── test_rlhf.py        # RLHF 训练测试
│   ├── test_rlvf.py        # RLVF 训练测试
│   ├── test_training.py    # 训练流程测试
│   ├── test_attention.py   # 注意力机制测试
│   ├── test_integration.py # 集成测试
│   └── ...                 # 更多测试文件
├── requirements-test.txt   # 测试依赖
├── pytest.ini              # pytest 配置
└── checkpoints/            # 模型检查点
    └── vocab.json          # BPE 词表 (含合并规则)
```

---

## 核心概念详解

### 阶段 0：分词器（Tokenizer）

**文件**：`tokenizer.py`

分词器是 LLM 的"翻译官"，将人类文本转换为模型可处理的数字序列。

#### 为什么需要分词

```
人类语言: "你好，世界"  ←── 模型无法直接理解
    ↓ 分词器
数字序列: [15, 23, 5, 89, 102]  ←── 模型可以处理
```

#### BPE 算法原理

```
┌─────────────────────────────────────────────────────────────┐
│                     BPE 分词器工作流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入: "你好，世界"                                         │
│         ↓                                                   │
│   Step 1: 字符切分                                          │
│         ["你", "好", "，", "世", "界"]                       │
│         ↓                                                   │
│   Step 2: 统计高频字符对，逐步合并                           │
│         ("你", "好") → 5 次 → 合并为 "你好"                  │
│         ("世", "界") → 3 次 → 合并为 "世界"                  │
│         ↓                                                   │
│   Step 3: 构建词表                                          │
│         ┌──────────────────────┐                           │
│         │  词表 (训练时构建)    │                           │
│         │  ──────────────────  │                           │
│         │  <pad> → 0           │                           │
│         │  <unk> → 1           │                           │
│         │  你    → 15          │                           │
│         │  好    → 23          │                           │
│         │  你好  → 100 (合并后) │                           │
│         │  世界  → 101 (合并后) │                           │
│         └──────────────────────┘                           │
│         ↓                                                   │
│   输出: [100, 5, 101]  (更短的序列!)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 特殊 Token

| Token | ID | 用途 |
|-------|-----|------|
| `<pad>` | 0 | 填充序列到相同长度 |
| `<unk>` | 1 | 表示未知字符 |
| `<bos>` | 2 | 标记序列开始 |
| `<eos>` | 3 | 标记序列结束 |
| `<\|im_start\|>` | 4 | 对话角色开始 |
| `<\|im_end\|>` | 5 | 对话角色结束 |

#### 对话格式（ChatML）

```
<|im_start|>user
你好，请介绍一下自己。<|im_end|>
<|im_start|>assistant
你好！我是 MyLLM，一个小型语言模型。<|im_end|>
```

---

### 模型架构：Transformer

**文件**：`model.py`

Transformer 是现代 LLM 的核心架构，本项目实现了 GPT 风格的 Decoder-Only 结构。

#### 整体结构

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer 架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入: Token IDs [15, 23, 5]                               │
│         ↓                                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Token Embedding                                    │   │
│   │  将每个 ID 映射为 256 维向量                          │   │
│   └─────────────────────────────────────────────────────┘   │
│         ↓ +                                                 │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Position Embedding (RoPE)                          │   │
│   │  添加位置信息                                        │   │
│   └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Transformer Block × 6 层                           │   │
│   │  ┌─────────────────────────────────────────────┐   │   │
│   │  │  LayerNorm                                  │   │   │
│   │  │       ↓                                     │   │   │
│   │  │  Multi-Head Self-Attention (8 heads)       │   │   │
│   │  │       ↓ + 残差连接                          │   │   │
│   │  │  LayerNorm                                  │   │   │
│   │  │       ↓                                     │   │   │
│   │  │  Feed-Forward Network                       │   │   │
│   │  │       ↓ + 残差连接                          │   │   │
│   │  └─────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Output Projection                                  │   │
│   │  [batch, seq, 256] → [batch, seq, vocab_size]      │   │
│   └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│   输出: 下一个 token 的概率分布                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Self-Attention 详解

Self-Attention 是 Transformer 的核心，让模型学会"关注"输入中的相关部分。

```
┌─────────────────────────────────────────────────────────────┐
│                   Self-Attention 机制                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   输入 X: "我 爱 学习"                                       │
│         ↓                                                   │
│   生成 Q, K, V:                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Q (Query) = X × W_q   "我想找什么信息？"            │   │
│   │  K (Key)   = X × W_k   "我有什么标签？"              │   │
│   │  V (Value) = X × W_v   "我的实际内容是什么？"         │   │
│   └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│   计算注意力分数:                                            │
│   Attention = softmax(Q × K^T / √d) × V                    │
│                                                             │
│   因果遮罩 (Causal Mask):                                   │
│   GPT 是自回归模型，只能看到之前的词                         │
│                                                             │
│       我    爱    学习                                       │
│  我   [✓]  [✗]   [✗]                                        │
│  爱   [✓]  [✓]   [✗]                                        │
│  学习 [✓]  [✓]   [✓]                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 模型配置

```python
from config import get_mini_config

config = get_mini_config()
# vocab_size:    2000    词表大小
# emb_dim:       256     嵌入维度
# num_heads:     8       注意力头数
# num_layers:    6       Transformer 层数
# context_size:  256     最大上下文长度
# 参数量:        ~5M
```

---

### 阶段 1：预训练（Pretrain）

**目标**：学习语言的基本规律，能够预测下一个词。

#### 训练方式

```
┌─────────────────────────────────────────────────────────────┐
│                      预训练过程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   训练数据: "人工智能是计算机科学的一个分支"                   │
│                                                             │
│   自回归训练 (预测下一个词):                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  输入: [人]           → 预测: 工                     │   │
│   │  输入: [人,工]        → 预测: 智                     │   │
│   │  输入: [人,工,智]     → 预测: 能                     │   │
│   │  输入: [人,工,智,能]  → 预测: 是                     │   │
│   │  ...                                                │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   损失函数: Cross-Entropy Loss                              │
│   L = -log(P(正确的下一个词))                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 预训练后的能力

- ✅ 理解语言结构和语法
- ✅ 学习词语之间的关联
- ✅ 能够续写文本
- ❌ 不会遵循指令
- ❌ 不会进行对话

---

### 阶段 2：SFT（监督微调）

**目标**：学习对话格式，获得指令遵循能力。

#### 训练方式

```
┌─────────────────────────────────────────────────────────────┐
│                       SFT 训练过程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   训练数据格式:                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  <|im_start|>user                                   │   │
│   │  什么是机器学习？<|im_end|>                          │   │
│   │  <|im_start|>assistant                              │   │
│   │  机器学习是人工智能的一个分支，让计算机从数据中         │   │
│   │  学习规律，而不需要明确编程。<|im_end|>               │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   训练目标:                                                  │
│   - 只计算 assistant 回复部分的 loss                        │
│   - 学习"看到用户问题后，生成合适的回复"                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### SFT 后的能力

- ✅ 预训练的所有能力
- ✅ 理解对话格式
- ✅ 遵循用户指令
- ✅ 生成有帮助的回复
- ❌ 可能生成不安全/不准确的内容

---

### 阶段 3：奖励模型（Reward Model）

**文件**：`reward_model.py`

**目标**：学习人类偏好，能够给回答打分。

#### 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                     奖励模型训练                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   训练数据 (偏好对):                                         │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Prompt: "什么是机器学习？"                          │   │
│   │                                                     │   │
│   │  Chosen (人类偏好):                                  │   │
│   │  "机器学习是AI的分支，让计算机从数据中学习规律..."     │   │
│   │                                                     │   │
│   │  Rejected (人类不偏好):                              │   │
│   │  "机器学习就是机器在学习，很简单的概念。"             │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   损失函数 (Bradley-Terry):                                  │
│   L = -log(σ(r_chosen - r_rejected))                        │
│                                                             │
│   目标: 让 r_chosen > r_rejected                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 阶段 4：RLHF（基于人类反馈的强化学习）

**文件**：`rlhf.py`

**目标**：利用奖励模型指导策略优化，生成更符合人类偏好的回答。

#### PPO 算法流程

```
┌─────────────────────────────────────────────────────────────┐
│                      RLHF (PPO) 训练                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   每个训练步骤:                                              │
│                                                             │
│   Step 1: 采样 ──► Policy Model 生成回答                     │
│   Step 2: 评分 ──► Reward Model 打分                        │
│   Step 3: 计算优势 ──► Advantage = reward - baseline        │
│   Step 4: PPO 更新 ──► 带裁剪的策略更新                      │
│                                                             │
│   PPO 损失 (带裁剪):                                        │
│   L = -min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)                 │
│                                                             │
│   裁剪作用: 防止策略更新过大导致训练不稳定                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### RLHF 后的能力

- ✅ 之前所有能力
- ✅ 生成更符合人类偏好的回答
- ✅ 更安全、更有帮助

---

### 阶段 5：RLVF（基于可验证反馈的强化学习）

**文件**：`rlvf.py`

**目标**：利用可自动验证的正确答案作为奖励信号，提升精确推理能力。

#### 与 RLHF 的区别

```
┌─────────────────────────────────────────────────────────────┐
│                    RLHF vs RLVF 对比                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   RLHF:                                                     │
│   Prompt → Model → Response → Reward Model → 分数 (主观)    │
│   适用: 开放式问题                                          │
│                                                             │
│   RLVF:                                                     │
│   Task → Model → Response → Verifier → 对/错 (客观)         │
│   适用: 有明确答案的问题                                     │
│   优点: 自动验证，无需人类标注                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 验证器类型

**1. 数学验证器 (MathVerifier)**
```python
prompt:   "计算 15 + 27 = ?"
expected: "42"

response: "答案是 42。"     → reward = +1.0 (正确)
response: "答案是 43。"     → reward = -0.5 (错误)
```

**2. 逻辑验证器 (LogicVerifier)**
```python
prompt:   "所有猫都是动物，小花是猫，小花是动物吗？"
expected: "是"

response: "是的，小花是动物。"  → reward = +1.0
response: "不一定。"           → reward = -0.5
```

---

## 使用指南

### 模型训练

```bash
# 完整 5 阶段训练
python3 train.py

# 跳过特定阶段
python3 train.py --skip-pretrain              # 跳过预训练
python3 train.py --skip-pretrain --skip-sft   # 只训练 RL 阶段
python3 train.py --skip-rlhf --skip-rlvf      # 只训练基础阶段

# 自定义训练参数
python3 train.py --pretrain_epochs 10 --sft_epochs 5 --batch_size 32
```

**阶段控制参数：**
- `--skip-pretrain` - 跳过预训练阶段
- `--skip-sft` - 跳过 SFT 阶段
- `--skip-reward` - 跳过奖励模型训练
- `--skip-rlhf` - 跳过 RLHF 阶段
- `--skip-rlvf` - 跳过 RLVF 阶段

**训练参数：**
- `--pretrain_epochs` - 预训练轮数（默认 10）
- `--sft_epochs` - SFT 轮数（默认 20）
- `--rlhf_episodes` - RLHF 轮数（默认 100）
- `--rlvf_iterations` - RLVF 迭代次数（默认 60）
- `--batch_size` - 批次大小（默认 16）

### 文本生成与对话

```bash
# 指定提示词生成文本
python3 generate.py --prompt "你好" --max_length 50

# 进入交互式对话模式
python3 generate.py --interactive

# 调整生成参数
python3 generate.py --prompt "人工智能" \
    --temperature 0.8 \
    --top_k 10 \
    --top_p 0.9 \
    --max_length 100

# 贪婪解码（确定性输出）
python3 generate.py --prompt "你好" --greedy

# 使用指定模型
python3 generate.py --checkpoint checkpoints/sft_final.pt --prompt "你好"

# 使用 LoRA 权重进行推理
python3 generate.py --interactive --lora checkpoints/lora/final
```

**生成参数说明：**
- `--temperature` - 温度参数，控制随机性（默认 0.8）
- `--top_k` - Top-k 采样（默认 10，0 表示不使用）
- `--top_p` - Top-p 采样（默认 0.9，1.0 表示不使用）
- `--max_length` - 最大生成长度（默认 100）
- `--repetition_penalty` - 重复惩罚系数（默认 1.2，>1.0 降低已生成 token 的概率）
- `--lora` - LoRA 权重路径（不指定则自动检测）
- `--greedy` - 使用贪婪解码

### 使用不同阶段的模型

```bash
# 使用预训练模型（只会续写文本，不会对话）
python3 generate.py --checkpoint checkpoints/pretrain_final.pt --prompt "人工智能"

# 使用 SFT 模型（具备基础对话能力）
python3 generate.py --interactive --checkpoint checkpoints/sft_final.pt

# 使用 RLHF 模型（回答更符合人类偏好）
python3 generate.py --interactive --checkpoint checkpoints/rlhf_final.pt

# 使用 RLVF 模型（完整训练，推理能力更强）
python3 generate.py --interactive --checkpoint checkpoints/rlvf_final.pt
```

**各阶段模型对比：**

| 模型 | 文件 | 能力 |
|------|------|------|
| Pretrain | `pretrain_final.pt` | 续写文本，无对话能力 |
| SFT | `sft_final.pt` | 基础对话，遵循指令 |
| RLHF | `rlhf_final.pt` | 回答更自然、更安全 |
| RLVF | `rlvf_final.pt` | 精确推理能力增强 |

### 单元测试

```bash
# 安装测试依赖
pip install -r requirements-test.txt

# 运行所有测试
python3 -m pytest tests/ -v

# 运行特定测试文件
python3 -m pytest tests/test_model.py -v
python3 -m pytest tests/test_tokenizer.py -v
python3 -m pytest tests/test_lora.py -v
python3 -m pytest tests/test_config.py -v
python3 -m pytest tests/test_generate.py -v
python3 -m pytest tests/test_reward_model.py -v

# 运行带覆盖率报告的测试
python3 -m pytest tests/ --cov=. --cov-report=term-missing

# 并行测试（需要 pytest-xdist）
python3 -m pytest tests/ -n auto
```

**测试覆盖模块：**

| 测试文件 | 覆盖模块 | 测试内容 |
|---------|---------|---------|
| `test_tokenizer.py` | `tokenizer.py` | BPE 分词、编码解码、保存加载 |
| `test_model.py` | `model.py` | GPT 架构、前向传播、梯度流 |
| `test_config.py` | `config.py` | 配置验证、预设配置、RLHF/RLVF 配置 |
| `test_lora.py` | `lora.py` | LoRA 层、权重合并、保存加载 |
| `test_generate.py` | `generate.py` | 文本生成、采样策略、困惑度 |
| `test_reward_model.py` | `reward_model.py` | 奖励模型、Bradley-Terry 损失 |
| `test_rlhf.py` | `rlhf.py` | PPO 训练、RLHF 流程 |
| `test_rlvf.py` | `rlvf.py` | RLVF 训练、可验证反馈 |
| `test_training.py` | `train.py` | 完整训练流程测试 |
| `test_attention.py` | `model.py` | 注意力机制、因果遮罩 |
| `test_integration.py` | 全模块 | 端到端集成测试 |

### 训练后的模型文件

```
checkpoints/
├── pretrain_final.pt   # 预训练模型
├── sft_final.pt        # SFT 模型
├── reward_model.pt     # 奖励模型
├── rlhf_final.pt       # RLHF 模型
└── rlvf_final.pt       # RLVF 模型
```

---

## 高级功能

### LoRA 高效微调

**文件**：`lora.py`, `train_lora.py`, `inference_lora.py`

LoRA（Low-Rank Adaptation）是一种高效的模型微调方法，只需训练不到 2% 的参数即可达到全量微调的效果。

#### 核心原理

```
┌─────────────────────────────────────────────────────────────┐
│                        LoRA 原理                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   原始模型:                                                  │
│   h = W · x        (W 是预训练权重，参数量大)                 │
│                                                             │
│   LoRA 修改:                                                 │
│   h = W · x + (B · A) · x                                   │
│       ─────   ─────────                                     │
│       冻结      可训练                                       │
│                                                             │
│   关键点:                                                    │
│   - W: 原始权重 (d_out × d_in)     → 冻结，不更新           │
│   - A: 低秩矩阵 (r × d_in)         → 可训练                │
│   - B: 低秩矩阵 (d_out × r)        → 可训练                │
│   - r << d_in, d_out  (例如 r=8, d=256)                    │
│                                                             │
│   效果: 原始参数 3.7M → 可训练参数 73K (约 2%)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### LoRA 训练

```bash
# 基础用法 - 使用 SFT 模型进行 LoRA 微调
python train_lora.py

# 自定义参数
python train_lora.py \
    --base_model checkpoints/sft_final.pt \
    --lora_r 8 \
    --lora_alpha 16 \
    --epochs 3 \
    --lr 1e-4

# 对更多层使用 LoRA（参数更多，效果可能更好）
python train_lora.py --target_modules c_attn c_proj linear1 linear2
```

#### LoRA 推理

```bash
# 使用 LoRA 模型进行交互式对话
python inference_lora.py

# 指定 LoRA 权重路径
python inference_lora.py checkpoints/lora/my_task
```

#### 代码中使用

```python
from model import GPT, GPTConfig
from lora import LoRAConfig, apply_lora_to_model, save_lora, load_lora

# 1. 加载基础模型
config = GPTConfig(vocab_size=2000, emb_dim=256, num_heads=8, num_layers=6)
model = GPT(config)

# 2. 配置并应用 LoRA
lora_config = LoRAConfig(
    r=8,                                # 秩
    alpha=16,                           # 缩放因子
    dropout=0.05,                       # Dropout
    target_modules=["c_attn", "c_proj"] # 目标层
)
model = apply_lora_to_model(model, lora_config)

# 3. 训练（只优化 LoRA 参数）
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)

# 4. 保存 LoRA 权重（仅几百 KB）
save_lora(model, "checkpoints/my_lora")

# 5. 加载 LoRA 到新模型
new_model = GPT(config)
new_model = load_lora(new_model, "checkpoints/my_lora")
```

#### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `r` | 低秩维度，越大能力越强 | 8-16 |
| `alpha` | 缩放因子，控制 LoRA 的影响程度 | 16-32 (通常 = 2×r) |
| `dropout` | LoRA 层的 Dropout 比例 | 0.05 |
| `target_modules` | 应用 LoRA 的层 | `["c_attn", "c_proj"]` |

#### LoRA vs 全量微调

| 对比项 | 全量微调 | LoRA |
|--------|----------|------|
| 可训练参数 | 100% | ~1-2% |
| 显存占用 | 高 | 低 |
| 训练速度 | 慢 | 快 |
| 存储空间 | 完整模型 | 仅 LoRA 权重 |
| 多任务 | 每任务一个模型 | 共享基座 + 多个 LoRA |

#### 训练后的文件

```
checkpoints/lora/
├── final/
│   ├── lora_weights.pt    # LoRA 权重（~300KB）
│   └── lora_config.json   # LoRA 配置
└── training_log.json      # 训练日志
```

---

## 附录

### 训练数据统计

| 数据类型 | 文件 | 数量 | 用途 |
|----------|------|------|------|
| SFT 对话 | `sft_data.json` | 94 条 | 对话能力训练 |
| 偏好数据 | `reward_data.json` | 40 对 | 奖励模型训练 |
| RLVF 任务 | `rlvf_data.json` | 40 条 | 精确推理训练 |

### 参考资料

- 《Build a Large Language Model (From Scratch)》
- "Attention Is All You Need" - Transformer 原论文
- "Training language models to follow instructions with human feedback" - InstructGPT/RLHF
- "Proximal Policy Optimization Algorithms" - PPO 论文
- "LoRA: Low-Rank Adaptation of Large Language Models" - LoRA 论文
- "Constitutional AI" - Anthropic

## 📚 术语表

这里整理了项目中出现的所有专业术语，帮助新人理解：

### 1. LLM (Large Language Model)
大语言模型，能够理解和生成人类语言的人工智能模型。

### 2. Transformer
现代 LLM 的核心架构，基于自注意力机制。

### 3. BPE (Byte Pair Encoding)
字节对编码，一种高效的子词分词算法。

### 4. Pretrain (Pre-training)
预训练，在大量无标注文本上训练模型，学习语言规律。

### 5. SFT (Supervised Fine-Tuning)
监督微调，在人工标注的对话数据上训练模型，获得指令遵循能力。

### 6. Reward Model
奖励模型，学习人类偏好，能够给回答打分。

### 7. RLHF (Reinforcement Learning from Human Feedback)
基于人类反馈的强化学习，利用奖励模型指导策略优化。

### 8. RLVF (Reinforcement Learning with Verified Feedback)
基于可验证反馈的强化学习，利用自动验证的正确答案作为奖励信号。

### 9. PPO (Proximal Policy Optimization)
近端策略优化，一种稳定的强化学习算法。

### 10. LoRA (Low-Rank Adaptation)
低秩适应，一种高效的模型微调方法。

### 11. Self-Attention
自注意力机制，让模型学会"关注"输入中的相关部分。

### 12. Causal Mask
因果遮罩，确保模型只能看到之前的词，无法看到未来的词。

## ❓ 常见问题解答 (FAQ)

这里整理了新人学习过程中可能遇到的问题和解决方案：

### 1. 安装依赖时遇到问题？
**解决方案**：
- 确保使用 Python 3.8+ 版本
- 使用虚拟环境安装：
  ```bash
  python3 -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate      # Windows
  pip install -r requirements.txt
  ```

### 2. 训练时内存不足？
**解决方案**：
- 降低 batch_size 参数：`python train.py --batch_size 8`
- 使用 CPU 训练（默认就是 CPU）
- 关闭其他占用内存的程序

### 3. 训练速度太慢？
**解决方案**：
- 减少训练轮数：`python train.py --pretrain_epochs 5`
- 使用更小的数据集：修改 `train.py` 中的数据加载部分
- 如果有 GPU，可以安装 PyTorch GPU 版本加速训练

### 4. 生成的文本质量不好？
**解决方案**：
- 增加训练轮数
- 使用更大的模型配置：修改 `config.py` 中的超参数
- 调整生成参数：提高 temperature 增加多样性，调整 top_k/top_p

### 5. 遇到 CUDA 相关的错误？
**解决方案**：
- 检查 PyTorch 是否安装了 GPU 版本
- 确保 CUDA 版本与 PyTorch 兼容
- 切换到 CPU 训练：`python train.py --device cpu`

---

### 许可证

MIT License

---

**祝学习愉快！**
