"""
LoRA (Low-Rank Adaptation) 实现

LoRA 是一种高效的模型微调方法，核心思想是：
1. 冻结预训练模型的权重
2. 在特定层（如注意力层的 Q、K、V、O 投影）添加可训练的低秩矩阵
3. 只训练这些低秩矩阵，大幅减少训练参数量

原理：
    原始: h = Wx
    LoRA: h = Wx + BAx

    其中 W 是冻结的预训练权重 (d_out x d_in)
    B 是低秩矩阵 (d_out x r)，初始化为 0
    A 是低秩矩阵 (r x d_in)，随机初始化
    r 是秩（rank），通常远小于 d_in 和 d_out

优点：
1. 参数量大幅减少（可能只有原来的 0.1%）
2. 训练速度快，显存占用低
3. 可以为不同任务保存不同的 LoRA 权重
4. 推理时可以将 LoRA 权重合并到原模型，无额外开销

参考：https://arxiv.org/abs/2106.09685
"""

import os
import json
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm


@dataclass
class LoRAConfig:
    """
    LoRA 配置类

    Attributes:
        r: 低秩矩阵的秩（rank），越大容量越强但参数越多
        alpha: 缩放因子，实际缩放为 alpha/r
        dropout: LoRA 层的 dropout 比例
        target_modules: 要应用 LoRA 的模块名称列表
        bias: 是否训练偏置，可选 "none", "all", "lora_only"

    对于 MyLLM 模型，可用的目标模块包括：
        - "c_attn": 注意力层的 QKV 投影（合并）
        - "c_proj": 注意力层的输出投影
        - "linear1": MLP 第一层
        - "linear2": MLP 第二层
    """
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    bias: str = "none"  # "none", "all", "lora_only"

    def to_dict(self) -> dict:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        return cls(**d)


class LoRALinear(nn.Module):
    """
    带 LoRA 适配器的线性层

    LoRA 的核心思想：低秩分解
    ==========================
    预训练模型的权重矩阵通常是"满秩"的，但微调时的更新往往是"低秩"的。
    这意味着我们可以用两个小矩阵的乘积来近似权重更新：

        ΔW ≈ B @ A

    其中：
        - W: 原始权重 (out_features × in_features)，例如 (768 × 768)
        - A: 降维矩阵 (r × in_features)，例如 (8 × 768)
        - B: 升维矩阵 (out_features × r)，例如 (768 × 8)
        - r: 秩（rank），通常取 4, 8, 16 等小值

    参数量对比（假设 in_features = out_features = 768, r = 8）：
        - 原始 W: 768 × 768 = 589,824 参数
        - LoRA (A+B): 8 × 768 + 768 × 8 = 12,288 参数
        - 减少比例: 98%

    数学公式：
        原始: y = Wx
        LoRA: y = Wx + BAx × (α/r)
                = Wx + ΔWx × scaling

    其中 α/r 是缩放因子，用于控制 LoRA 的影响强度。

    维度可视化：
    ┌─────────────────────────────────────────────────────────────┐
    │                                                              │
    │   输入 x        A 矩阵         中间态 z       B 矩阵         输出 Δy     │
    │  ┌───┐       ┌───────┐        ┌───┐       ┌───────┐        ┌───┐   │
    │  │   │       │       │        │   │       │       │        │   │   │
    │  │ d │  ──→  │ r × d │  ──→   │ r │  ──→  │ d × r │  ──→   │ d │   │
    │  │   │       │       │        │   │       │       │        │   │   │
    │  └───┘       └───────┘        └───┘       └───────┘        └───┘   │
    │  (d,)         (r, d)          (r,)         (d, r)          (d,)    │
    │                                                                     │
    │  d = in/out_features (如 768)                                       │
    │  r = rank (如 8)                                                    │
    │                                                                     │
    │  计算: z = A @ x    (768→8，降维)                                   │
    │        Δy = B @ z   (8→768，升维)                                   │
    └─────────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        """
        初始化 LoRA 层

        Args:
            original_layer: 原始的 nn.Linear 层
            r: 低秩矩阵的秩（rank）
               - 越大：表达能力越强，但参数越多
               - 越小：参数更少，但可能欠拟合
               - 推荐值：4, 8, 16, 32
            alpha: 缩放因子
               - 控制 LoRA 更新的强度
               - 实际缩放 = alpha / r
               - 通常设为 r 的 1-2 倍
            dropout: dropout 比例，用于正则化
        """
        super().__init__()

        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r  # 缩放因子，通常为 1.0 或 2.0

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # ============================================================
        # 冻结原始权重（LoRA 的核心：只训练新增的小矩阵）
        # ============================================================
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # ============================================================
        # 创建 LoRA 矩阵
        # ============================================================
        # A 矩阵: (r × in_features) - 将输入从 in_features 降维到 r
        # B 矩阵: (out_features × r) - 将中间表示从 r 升维到 out_features
        #
        # 初始化策略的关键设计：
        #   - A: 使用 Kaiming 初始化（保持方差稳定）
        #   - B: 使用零初始化
        #
        # 为什么 B 初始化为零？
        #   训练开始时：ΔW = B @ A = 0 @ A = 0
        #   这保证了：
        #   1. 初始状态下 LoRA 不改变原模型的输出
        #   2. 训练从原模型的良好状态开始
        #   3. 避免随机初始化破坏预训练的知识
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Kaiming 初始化 A 矩阵
        # a=sqrt(5) 是 PyTorch Linear 层的默认值，保持与原始层一致
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Dropout（可选的正则化）
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 合并状态标记
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        计算: y = Wx + BAx × scaling

        实现细节：
        ---------
        我们不直接计算 (B @ A) @ x，而是分两步：
            1. z = A @ x   (先降维)
            2. Δy = B @ z  (再升维)

        这样更高效，因为避免了计算 B @ A 这个大矩阵。

        计算量对比（假设 batch=32, seq=128, d=768, r=8）：
            方法1: (B @ A) @ x
                   B @ A: 768 × 8 × 768 = 4.7M 次乘法
                   结果 @ x: 768 × 768 × (32×128) = 2.4B 次乘法
                   总计: ~2.4B

            方法2: B @ (A @ x)
                   A @ x: 8 × 768 × (32×128) = 25M 次乘法
                   B @ 结果: 768 × 8 × (32×128) = 25M 次乘法
                   总计: ~50M

            方法2 快 ~48 倍！
        """
        if self.merged:
            # 如果已合并，直接使用原始层（无额外计算）
            return self.original_layer(x)

        # 原始输出: y = Wx
        original_output = self.original_layer(x)

        # ============================================================
        # LoRA 输出: Δy = B @ (A @ x) × scaling
        # ============================================================
        # 注意：F.linear(x, W) 计算的是 x @ W^T（即 W @ x 的效果）
        lora_output = self.dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # z = x @ A^T，形状从 (..., d) → (..., r)
        lora_output = F.linear(lora_output, self.lora_B)  # Δy = z @ B^T，形状从 (..., r) → (..., d)
        lora_output = lora_output * self.scaling

        # 最终输出 = 原始输出 + LoRA 增量
        return original_output + lora_output

    def merge_weights(self):
        """
        将 LoRA 权重合并到原始权重中

        合并公式: W' = W + B @ A × scaling

        为什么是 B @ A 而不是 A @ B？
        ---------------------------
        看维度就清楚了：
            - A: (r × in_features)，如 (8 × 768)
            - B: (out_features × r)，如 (768 × 8)
            - B @ A: (768 × 8) @ (8 × 768) = (768 × 768) ✓
            - A @ B: (8 × 768) @ (768 × 8) = (8 × 8) ✗

        B @ A 的结果维度 (out_features × in_features) 与原始权重 W 相同。

        可视化：
        ┌─────────┐     ┌───────┐     ┌─────────┐
        │  B      │  @  │   A   │  =  │  ΔW     │
        │ (d × r) │     │(r × d)│     │ (d × d) │
        │  768×8  │     │ 8×768 │     │ 768×768 │
        └─────────┘     └───────┘     └─────────┘

        合并的优点：
            1. 推理时无额外计算开销
            2. 可以将 LoRA 导出为普通模型权重

        合并的缺点：
            1. 无法继续训练这个 LoRA
            2. 无法轻松切换到其他 LoRA
        """
        if not self.merged:
            # 计算权重增量: ΔW = B @ A × scaling
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            # 合并到原始权重: W' = W + ΔW
            self.original_layer.weight.data += delta_weight
            self.merged = True

    def unmerge_weights(self):
        """
        从原始权重中移除 LoRA 权重

        公式: W = W' - B @ A × scaling

        使用场景：
            1. 需要继续训练 LoRA
            2. 需要切换到不同的 LoRA 权重
            3. 需要比较有无 LoRA 的效果
        """
        if self.merged:
            # 计算并减去权重增量
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data -= delta_weight
            self.merged = False

    def get_lora_params(self) -> Dict[str, torch.Tensor]:
        """获取 LoRA 参数"""
        return {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data
        }

    def set_lora_params(self, params: Dict[str, torch.Tensor]):
        """设置 LoRA 参数"""
        self.lora_A.data = params["lora_A"]
        self.lora_B.data = params["lora_B"]


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = True
) -> nn.Module:
    """
    将 LoRA 应用到模型的指定层

    Args:
        model: 要修改的模型
        config: LoRA 配置
        verbose: 是否打印信息

    Returns:
        修改后的模型（原地修改）
    """
    if verbose:
        print("=" * 50)
        print("应用 LoRA 到模型")
        print("=" * 50)
        print(f"  秩 (r): {config.r}")
        print(f"  缩放因子 (alpha): {config.alpha}")
        print(f"  Dropout: {config.dropout}")
        print(f"  目标模块: {config.target_modules}")

    lora_layers = []

    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 遍历所有模块
    for name, module in model.named_modules():
        # 检查是否匹配目标模块
        for target in config.target_modules:
            if target in name and isinstance(module, nn.Linear):
                # 找到父模块
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]

                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model

                # 创建 LoRA 层
                lora_layer = LoRALinear(
                    original_layer=module,
                    r=config.r,
                    alpha=config.alpha,
                    dropout=config.dropout
                )

                # 替换原始层
                setattr(parent, child_name, lora_layer)
                lora_layers.append(name)

    if verbose:
        print(f"\n已应用 LoRA 的层 ({len(lora_layers)} 个):")
        for layer_name in lora_layers:
            print(f"  - {layer_name}")

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        print(f"  可训练比例: {trainable_params/total_params*100:.2f}%")
        print("=" * 50)

    # 保存 LoRA 配置到模型
    model.lora_config = config
    model.lora_layers = lora_layers

    return model


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    获取模型中所有 LoRA 参数

    Args:
        model: 应用了 LoRA 的模型

    Returns:
        包含所有 LoRA 参数的字典
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data.clone()
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data.clone()

    return lora_state_dict


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """
    加载 LoRA 参数到模型

    Args:
        model: 应用了 LoRA 的模型
        state_dict: LoRA 参数字典
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"

            if a_key in state_dict and b_key in state_dict:
                module.lora_A.data = state_dict[a_key].clone()
                module.lora_B.data = state_dict[b_key].clone()


def save_lora(
    model: nn.Module,
    path: str,
    config: Optional[LoRAConfig] = None
):
    """
    保存 LoRA 权重和配置

    Args:
        model: 应用了 LoRA 的模型
        path: 保存路径（目录）
        config: LoRA 配置（如果为 None，使用模型中保存的配置）
    """
    os.makedirs(path, exist_ok=True)

    # 保存 LoRA 权重
    lora_state_dict = get_lora_state_dict(model)
    torch.save(lora_state_dict, os.path.join(path, "lora_weights.pt"))

    # 保存配置
    config = config or getattr(model, "lora_config", None)
    if config:
        with open(os.path.join(path, "lora_config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    print(f"LoRA 已保存到: {path}")


def load_lora(
    model: nn.Module,
    path: str,
    apply_if_needed: bool = True
) -> nn.Module:
    """
    加载 LoRA 权重和配置

    Args:
        model: 基础模型
        path: 保存路径（目录）
        apply_if_needed: 如果模型还没有应用 LoRA，是否自动应用

    Returns:
        加载了 LoRA 的模型
    """
    # 加载配置
    config_path = os.path.join(path, "lora_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = LoRAConfig.from_dict(json.load(f))
    else:
        raise ValueError(f"找不到 LoRA 配置文件: {config_path}")

    # 如果模型还没有应用 LoRA，先应用
    if apply_if_needed and not hasattr(model, "lora_layers"):
        model = apply_lora_to_model(model, config, verbose=False)

    # 加载权重
    weights_path = os.path.join(path, "lora_weights.pt")
    state_dict = torch.load(weights_path, map_location="cpu")
    load_lora_state_dict(model, state_dict)

    print(f"LoRA 已从 {path} 加载")

    return model


def merge_lora(model: nn.Module):
    """
    将所有 LoRA 权重合并到原始模型

    合并后模型推理更快，但无法继续训练 LoRA
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
    print("LoRA 权重已合并到模型")


def unmerge_lora(model: nn.Module):
    """
    从模型中移除已合并的 LoRA 权重
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()
    print("LoRA 权重已从模型中移除")


class LoRATrainer:
    """
    LoRA 微调训练器

    用于使用 LoRA 对模型进行高效微调
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "auto"
    ):
        """
        初始化训练器

        Args:
            model: 基础模型
            lora_config: LoRA 配置
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 设备
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 应用 LoRA
        self.model = apply_lora_to_model(model, lora_config)
        self.model = self.model.to(self.device)

        self.lora_config = lora_config

        # 只优化 LoRA 参数
        lora_params = []
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
                lora_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )

        print(f"LoRA 训练器初始化完成")
        print(f"  设备: {self.device}")
        print(f"  学习率: {learning_rate}")
        print(f"  可训练参数: {sum(p.numel() for p in lora_params):,}")

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0
    ) -> float:
        """
        训练一个 epoch

        Args:
            dataloader: 数据加载器，返回 (input_ids, target_ids)
            epoch: 当前 epoch 数

        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"LoRA Epoch {epoch + 1}")

        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # 前向传播
            _, loss = self.model(input_ids, target_ids)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )

            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(dataloader)

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 3,
        save_path: Optional[str] = None
    ) -> List[float]:
        """
        完整训练流程

        Args:
            dataloader: 数据加载器
            epochs: 训练轮数
            save_path: 保存路径（可选）

        Returns:
            每个 epoch 的损失列表
        """
        losses = []

        for epoch in range(epochs):
            loss = self.train_epoch(dataloader, epoch)
            losses.append(loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

        if save_path:
            save_lora(self.model, save_path, self.lora_config)

        return losses

    def save(self, path: str):
        """保存 LoRA 权重"""
        save_lora(self.model, path, self.lora_config)

    def get_model(self, merge: bool = False) -> nn.Module:
        """
        获取训练后的模型

        Args:
            merge: 是否合并 LoRA 权重到原始模型

        Returns:
            模型
        """
        if merge:
            merge_lora(self.model)
        return self.model


def demo_lora():
    """
    LoRA 使用示例
    """
    from model import GPT, GPTConfig

    print("=" * 60)
    print("LoRA 演示")
    print("=" * 60)

    # 1. 创建基础模型
    print("\n1. 创建基础模型")
    config = GPTConfig(
        vocab_size=1000,
        emb_dim=256,
        num_heads=8,
        num_layers=6,
        context_size=128
    )
    model = GPT(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   基础模型参数量: {total_params:,}")

    # 2. 配置 LoRA
    print("\n2. 配置 LoRA")
    lora_config = LoRAConfig(
        r=8,
        alpha=16,
        dropout=0.05,
        target_modules=["c_attn", "c_proj"]  # MyLLM 模型的注意力层
    )
    print(f"   秩: {lora_config.r}")
    print(f"   目标模块: {lora_config.target_modules}")

    # 3. 应用 LoRA
    print("\n3. 应用 LoRA")
    model = apply_lora_to_model(model, lora_config)

    # 4. 测试前向传播
    print("\n4. 测试前向传播")
    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))

    logits, loss = model(x, y)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {logits.shape}")
    print(f"   损失: {loss.item():.4f}")

    # 5. 保存和加载
    print("\n5. 保存和加载 LoRA")
    save_lora(model, "checkpoints/lora_demo", lora_config)

    # 创建新模型并加载 LoRA
    new_model = GPT(config)
    new_model = load_lora(new_model, "checkpoints/lora_demo")

    # 6. 合并权重
    print("\n6. 合并 LoRA 权重")
    merge_lora(new_model)

    # 验证输出一致
    with torch.no_grad():
        logits2, _ = new_model(x, y)

    diff = (logits - logits2).abs().max().item()
    print(f"   合并前后输出差异: {diff:.6f}")

    print("\n" + "=" * 60)
    print("LoRA 演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo_lora()
