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

    实现公式: output = W @ x + (B @ A) @ x * (alpha / r)

    其中:
    - W: 原始冻结的权重矩阵
    - A: 低秩矩阵 (r x in_features)，用 kaiming 初始化
    - B: 低秩矩阵 (out_features x r)，用零初始化
    - alpha/r: 缩放因子
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
            r: 低秩矩阵的秩
            alpha: 缩放因子
            dropout: dropout 比例
        """
        super().__init__()

        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # 冻结原始权重
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # 创建 LoRA 矩阵
        # A: (r, in_features) - 用 kaiming 初始化
        # B: (out_features, r) - 用零初始化（保证初始时 LoRA 不改变输出）
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化 A 矩阵
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 是否已合并
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        如果已合并，直接使用原始层
        否则，计算 original_output + lora_output
        """
        if self.merged:
            return self.original_layer(x)

        # 原始输出
        original_output = self.original_layer(x)

        # LoRA 输出: (B @ A) @ x * scaling
        # 分步计算更高效: B @ (A @ x)
        lora_output = self.dropout(x)
        lora_output = F.linear(lora_output, self.lora_A)  # x @ A^T
        lora_output = F.linear(lora_output, self.lora_B)  # (x @ A^T) @ B^T
        lora_output = lora_output * self.scaling

        return original_output + lora_output

    def merge_weights(self):
        """
        将 LoRA 权重合并到原始权重中

        合并后推理更快，但无法继续训练 LoRA
        """
        if not self.merged:
            # W' = W + B @ A * scaling
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data += delta_weight
            self.merged = True

    def unmerge_weights(self):
        """
        从原始权重中移除 LoRA 权重

        用于继续训练或切换不同的 LoRA
        """
        if self.merged:
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
