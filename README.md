# 从零手搓大模型 - 学习项目

一个完整的大语言模型学习项目，从零实现 GPT 风格的语言模型。

## 项目简介

本项目旨在帮助开发者理解大语言模型的底层原理，通过亲手实现每个组件，打破对大模型的神秘感。

### 核心内容

- **Tokenizer（分词器）**: BPE 算法实现
- **Embedding（嵌入层）**: Token 和位置嵌入
- **Attention（注意力机制）**: 自注意力和多头注意力
- **Transformer Block**: 完整的 Transformer 块
- **GPT Model**: 完整的生成式预训练模型
- **Training**: 模型训练流程
- **Generation**: 文本生成与采样策略

## 项目结构

```
my_llm/
├── data/               # 数据目录
│   └── tokenizer.json  # 训练好的分词器
├── checkpoints/        # 模型检查点
├── logs/              # 训练日志
├── tokenizer.py       # BPE 分词器实现
├── model.py           # GPT 模型组件实现
├── train.py           # 训练脚本
├── generate.py        # 文本生成/推理脚本
├── main.py            # 主入口
├── requirements.txt   # 依赖包
└── README.md          # 本文件
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 查看演示

```bash
# 运行完整演示（推荐先运行这个）
python main.py --mode demo

# 运行前向传播详细演示（强烈推荐！用具体数字展示每一步）
python model.py
```

这将依次展示：
- 分词器的训练和使用
- 模型各组件的输出
- **前向传播的每一步数据变化（带具体数字）**
- 一个小模型的完整训练和生成过程

### 2. 训练模型

```bash
# 使用默认参数训练
python main.py --mode train

# 使用自定义参数训练
python main.py --mode train \
    --vocab_size 1000 \
    --emb_dim 256 \
    --num_heads 8 \
    --num_layers 6 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3
```

### 3. 生成文本

```bash
# 生成文本
python main.py --mode generate --prompt "你好"

# 自定义生成参数
python main.py --mode generate \
    --prompt "人工智能" \
    --max_length 50 \
    --temperature 0.8 \
    --top_k 10 \
    --top_p 0.9
```

### 4. 交互式对话

```bash
# 进入交互模式
python main.py --mode interactive
```

---

## 核心概念详解

### 1. Token Embedding（词嵌入）

**一句话解释**：把整数ID变成向量（一组数字）

```
词表: ["我", "爱", "AI", "学", "习"]
       ID=0  ID=1  ID=2  ID=3  ID=4

嵌入表 (5个词 × 4维):
  ID=0 "我" → [0.1, 0.2, 0.3, 0.4]
  ID=1 "爱" → [0.5, 0.6, 0.7, 0.8]
  ...

输入 "我 爱" → [0, 1]
      ↓ 查表
输出 [[0.1, 0.2, 0.3, 0.4],   ← "我"的向量
      [0.5, 0.6, 0.7, 0.8]]   ← "爱"的向量
```

**关键点**：
- 本质是一个查找表（矩阵），用ID取对应的行
- 初始值是随机的，通过训练学习得到
- 训练后，相似的词会有相似的向量

---

### 2. Position Embedding（位置嵌入）

**一句话解释**：告诉模型每个词在第几个位置

```
位置嵌入表:
  位置0 → [0.01, 0.02, 0.01, 0.02]
  位置1 → [0.03, 0.04, 0.03, 0.04]
  ...

最终嵌入 = Token嵌入 + 位置嵌入
```

**为什么需要**：Transformer 本身看不到词的顺序，需要显式告诉它。

---

### 3. Self-Attention（自注意力）

**一句话解释**：让每个词看看其他词，决定该关注谁

```
核心公式：Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

步骤：
1. 生成 Q(Query), K(Key), V(Value)
2. Q @ K^T 计算相似度（注意力分数）
3. Softmax 归一化成概率
4. 用概率加权求和 V

因果Mask：当前词只能看之前的词，看不到未来
```

**形象理解**：
- Q = "我想要什么信息"
- K = "我有什么标签"
- V = "我的实际内容"
- 注意力分数 = Q和K的相似度

---

### 4. Transformer Block

**一句话解释**：让词之间"交流"并"思考"的处理单元

```
┌─────────────────────────────────┐
│  LayerNorm                      │
│  ↓                              │
│  Self-Attention (词间交流)       │
│  ↓                              │
│  + 残差连接                      │
├─────────────────────────────────┤
│  LayerNorm                      │
│  ↓                              │
│  Feed-Forward (独立思考)         │
│  ↓                              │
│  + 残差连接                      │
└─────────────────────────────────┘
```

**多层堆叠**：
- 1层 = 浅层理解
- 6层 = 深层理解
- 96层 = GPT-3级别理解

---

### 5. 前向传播完整流程

```
输入 "我 爱" [0, 1]
      ↓
┌─────────────────────────────────┐
│ 1. Token Embedding              │   [1,2] → [1,2,256]
│    ID → 向量 (查表)              │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ 2. Position Embedding           │   + [2,256]
│    加上位置信息                  │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ 3. Transformer Block × N        │   [1,2,256] → [1,2,256]
│    - Self-Attention: 词间交流   │
│    - Feed-Forward: 深度处理     │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ 4. LayerNorm                    │   标准化输出
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ 5. 输出投影 (lm_head)           │   [1,2,256] → [1,2,vocab_size]
│    向量 → 词表概率              │
└─────────────────────────────────┘
      ↓
Softmax → 预测下一个词
```

**运行 `python model.py` 可以看到每一步的具体数字！**

---

### 6. 反向传播与训练

**一句话解释**：从结果往回推，找出每个参数该怎么调

```
前向传播：输入 → 模型 → 预测结果
              ↓
         对比正确答案 → 计算误差(loss)
              ↓
反向传播：误差往回传 → 计算每个参数的梯度
              ↓
梯度下降：调整参数（包括 Embedding 表）
```

**代码中的体现**：
```python
# 1. 前向传播
logits, loss = model(input_ids, targets)

# 2. 反向传播（计算梯度）
loss.backward()

# 3. 更新参数
optimizer.step()
```

**为什么相似的词会有相似的向量**：
- "猫"和"狗"经常出现在相似的位置（"我喜欢___"）
- 为了预测对，模型让它们的向量变得相似
- 这是自动学习的，不是人为设计的

---

### 7. 文本生成（推理）

**自回归生成**：每次预测一个词，然后把它加入输入，继续预测

```
第1轮: 输入 [我] → 预测 "喜欢"
第2轮: 输入 [我, 喜欢] → 预测 "学习"
第3轮: 输入 [我, 喜欢, 学习] → 预测 "AI"
...循环...
```

**采样策略**：
| 策略 | 说明 |
|-----|------|
| Greedy | 总是选概率最高的词（确定但单调）|
| Temperature | 调整随机性（<1保守，>1随机）|
| Top-k | 只从前k个高概率词中选 |
| Top-p | 从累积概率达到p的词中选 |

---

## 代码详解

### 1. Tokenizer (`tokenizer.py`)

BPE（Byte Pair Encoding）分词器的完整实现：

```python
from tokenizer import BPETokenizer

# 训练分词器
tokenizer = BPETokenizer(vocab_size=1000)
tokenizer.fit(train_texts)

# 编码和解码
token_ids = tokenizer.encode("你好 世界")
text = tokenizer.decode(token_ids)

# 保存/加载
tokenizer.save("data/tokenizer.json")
tokenizer = BPETokenizer.load("data/tokenizer.json")
```

### 2. 模型结构 (`model.py`)

```python
from model import GPT, GPTConfig

# 创建配置
config = GPTConfig(
    vocab_size=1000,      # 词表大小
    emb_dim=256,          # 嵌入维度
    num_heads=8,          # 注意力头数
    num_layers=6,         # Transformer层数
    context_size=256,     # 最大上下文长度
    dropout=0.1           # Dropout比例
)

# 创建模型
model = GPT(config)

# 前向传播
logits, loss = model(input_ids, targets)
```

### 3. 训练 (`train.py`)

```python
# 训练循环核心代码
for epoch in range(epochs):
    for input_ids, target_ids in dataloader:
        # 前向传播
        logits, loss = model(input_ids, target_ids)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()
```

### 4. 生成 (`generate.py`)

```python
from generate import TextGenerator

generator = TextGenerator(model, tokenizer, device)

# 生成文本
output = generator.generate(
    prompt="你好",
    max_length=50,
    temperature=0.8,
    top_k=10,
    top_p=0.9
)
```

---

## 参数说明

### 模型参数

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| vocab_size | 词表大小 | 1000 | 越大表达能力越强 |
| emb_dim | 嵌入维度 | 256 | 越大模型越强但越慢 |
| num_heads | 注意力头数 | 8 | 必须能被emb_dim整除 |
| num_layers | Transformer层数 | 6 | 越多理解越深 |
| context_size | 最大上下文长度 | 256 | 一次能看多少词 |
| dropout | Dropout比例 | 0.1 | 防止过拟合 |

### 生成参数

| 参数 | 说明 | 范围 | 效果 |
|------|------|------|------|
| temperature | 控制随机性 | 0.1~2.0 | 小=保守，大=随机 |
| top_k | Top-k采样 | 0~100 | 只从前k个词选 |
| top_p | Top-p采样 | 0.0~1.0 | 从累积p概率的词选 |

---

## 学习路径

### 阶段一：理解分词器

1. 阅读 `tokenizer.py` 中的 `BPETokenizer` 类
2. 理解 BPE 算法的核心逻辑
3. 运行演示，观察分词结果

### 阶段二：理解前向传播

1. **运行 `python model.py`，观察详细的前向传播过程**
2. 阅读 `model.py`，按以下顺序：
   - `LayerNorm`: 层归一化
   - `FeedForward`: 前馈网络
   - `CausalSelfAttention`: 因果自注意力
   - `TransformerBlock`: Transformer 块
   - `GPT`: 完整模型

### 阶段三：理解训练

1. 阅读 `train.py` 中的数据准备逻辑
2. 理解损失函数的计算方式
3. 观察训练过程中损失的变化

### 阶段四：理解生成

1. 阅读 `generate.py` 中的采样策略
2. 尝试不同的 temperature、top_k、top_p 参数
3. 观察参数对生成结果的影响

---

## 常见问题

### Q: 为什么相似的词会有相似的向量？

因为它们经常出现在相似的上下文中。模型为了预测准确，会自动让它们的向量变得相似。

### Q: Embedding表的数字从哪来？

初始是随机生成的，通过训练（反向传播+梯度下降）自动学习得到。

### Q: 什么是因果Mask？

让当前词只能看到之前的词，看不到未来的词。这是语言模型生成时的必要约束。

### Q: LayerNorm有什么用？

把数值标准化到正常范围，防止数值爆炸或消失，让训练更稳定。

---

## 扩展建议

1. **使用真实数据集**：替换 `create_sample_data()` 中的示例数据
2. **增加模型规模**：尝试增加层数、维度，观察效果变化
3. **实现新的采样策略**：如 beam search
4. **添加评估指标**：困惑度（Perplexity）、BLEU 等
5. **实现 SFT（监督微调）**：添加对话格式的训练数据

---

## 参考资料

- 《Build a Large Language Model (From Scratch)》
- "Attention Is All You Need" (Transformer 原论文)
- "Language Models are Few-Shot Learners" (GPT-3 论文)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

---

## 许可证

MIT License

---

**祝学习愉快！如有问题，欢迎交流讨论。**
