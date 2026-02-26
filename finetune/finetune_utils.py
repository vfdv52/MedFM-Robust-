#!/usr/bin/env python3
"""
微调工具模块 - 完整修复版
包含详细诊断输出，运行后可以知道问题在哪里

【修复内容】
1. warmup_epochs 从 1 改为 0.1（减少warmup时间）
2. learning_rate 默认值从 1e-4 改为 5e-4
3. SegmentationLoss 增加数值稳定性处理

【新增内容 - LoRA+】
4. 添加 lora_plus_encoder 和 lora_plus_decoder 策略
5. LoRA+ 对 A/B 矩阵使用不同学习率
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class FinetuneStrategy(Enum):
    DECODER_ONLY = "decoder_only"
    DECODER_PROMPT = "decoder_prompt"
    ADAPTER_ONLY = "adapter_only"
    ENCODER_PARTIAL = "encoder_partial"
    ENCODER_FULL = "encoder_full"
    FULL = "full"
    LORA = "lora"
    # 新增 LoRA+ 策略
    LORA_PLUS_ENCODER = "lora_plus_encoder"  # LoRA+ 应用于 encoder，同时常规训练 decoder
    LORA_PLUS_DECODER = "lora_plus_decoder"  # LoRA+ 应用于 decoder

@dataclass
class FinetuneConfig:
    strategy: str = "decoder_only"
    learning_rate: float = 5e-4  # 【修复】从1e-4提高到5e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 4
    encoder_lr_scale: float = 0.1
    unfreeze_encoder_layers: int = 4  # 默认解冻4层
    lora_r: int = 8  # 统一默认值，与 argparse 一致
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    # 【新增】LoRA+ 特有参数
    lora_plus_lr_ratio: float = 16.0  # B矩阵学习率 = A矩阵学习率 * ratio (LoRA+ 论文推荐 16)
    warmup_epochs: float = 0.1  # 【修复】从1改为0.1，只warmup 10%的第一个epoch
    save_interval: int = 1
    eval_interval: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    output_dir: str = "./finetune_output"
    checkpoint_name: str = "finetuned_model.pth"
    
    def to_dict(self) -> Dict:
        return {
            "strategy": self.strategy,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "encoder_lr_scale": self.encoder_lr_scale,
            "unfreeze_encoder_layers": self.unfreeze_encoder_layers,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "lora_plus_lr_ratio": self.lora_plus_lr_ratio,  # 【新增】
            "warmup_epochs": self.warmup_epochs,
            "use_amp": self.use_amp,
            "output_dir": self.output_dir,
        }


def freeze_module(module: nn.Module):
    """冻结模块"""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module):
    """解冻模块"""
    for param in module.parameters():
        param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def print_trainable_parameters(model: nn.Module, name: str = "Model"):
    trainable, total = count_parameters(model)
    pct = 100 * trainable / total if total > 0 else 0
    print(f"\n{'='*70}")
    print(f"📊 {name} 参数统计:")
    print(f"   可训练参数: {trainable:,}")
    print(f"   总参数:     {total:,}")
    print(f"   可训练比例: {pct:.2f}%")
    print(f"{'='*70}")


def diagnose_model_structure(model, model_name):
    """诊断模型结构 - 关键诊断函数"""
    print(f"\n{'='*70}")
    print(f"🔬 诊断 {model_name} 模型结构")
    print(f"{'='*70}")
    
    # 获取内部SAM模型
    if hasattr(model, 'model'):
        sam = model.model
    else:
        sam = model
    
    print(f"\n📦 顶层模块:")
    for name, child in sam.named_children():
        param_count = sum(p.numel() for p in child.parameters())
        print(f"   {name}: {type(child).__name__} ({param_count:,} 参数)")
    
    # 检查encoder结构
    if hasattr(sam, 'vision_encoder'):
        encoder = sam.vision_encoder
        print(f"\n📦 vision_encoder 结构:")
        for name, child in encoder.named_children():
            if isinstance(child, nn.ModuleList):
                print(f"   {name}: ModuleList (长度={len(child)})")
            else:
                print(f"   {name}: {type(child).__name__}")
        
        # 检查layers属性
        print(f"\n🔍 encoder层属性检查:")
        for attr in ['layers', 'blocks', 'layer', 'encoder_layers']:
            if hasattr(encoder, attr):
                obj = getattr(encoder, attr)
                if isinstance(obj, nn.ModuleList):
                    print(f"   ✅ encoder.{attr} 存在, 长度={len(obj)}")
                else:
                    print(f"   ⚠️ encoder.{attr} 存在但不是ModuleList: {type(obj)}")
            else:
                print(f"   ❌ encoder.{attr} 不存在")


def diagnose_trainable_params(model, strategy):
    """诊断可训练参数分布"""
    print(f"\n{'='*70}")
    print(f"🔬 诊断可训练参数 (策略: {strategy})")
    print(f"{'='*70}")
    
    if hasattr(model, 'model'):
        sam = model.model
    else:
        sam = model
    
    # 按模块统计
    module_stats = {}
    for name, param in sam.named_parameters():
        parts = name.split('.')
        module = parts[0] if parts else 'other'
        
        if module not in module_stats:
            module_stats[module] = {'total': 0, 'trainable': 0}
        
        module_stats[module]['total'] += param.numel()
        if param.requires_grad:
            module_stats[module]['trainable'] += param.numel()
    
    print(f"\n📊 各模块可训练参数:")
    for module, stats in sorted(module_stats.items()):
        pct = 100 * stats['trainable'] / stats['total'] if stats['total'] > 0 else 0
        status = "✅" if stats['trainable'] > 0 else "❌"
        print(f"   {status} {module}: {stats['trainable']:,} / {stats['total']:,} ({pct:.1f}%)")
    
    # 验证策略是否正确执行
    print(f"\n🔍 策略验证:")
    
    if strategy == "decoder_only":
        if module_stats.get('mask_decoder', {}).get('trainable', 0) == 0:
            print(f"   ❌ 错误: mask_decoder 应该可训练但实际不是!")
        else:
            print(f"   ✅ mask_decoder 已解冻")
        if module_stats.get('vision_encoder', {}).get('trainable', 0) > 0:
            print(f"   ⚠️ 警告: vision_encoder 不应该被训练!")
            
    elif strategy == "decoder_prompt":
        if module_stats.get('mask_decoder', {}).get('trainable', 0) == 0:
            print(f"   ❌ 错误: mask_decoder 应该可训练!")
        else:
            print(f"   ✅ mask_decoder 已解冻")
        if module_stats.get('prompt_encoder', {}).get('trainable', 0) == 0:
            print(f"   ❌ 错误: prompt_encoder 应该可训练!")
        else:
            print(f"   ✅ prompt_encoder 已解冻")
            
    elif strategy == "encoder_partial":
        if module_stats.get('vision_encoder', {}).get('trainable', 0) == 0:
            print(f"   ❌ 错误: vision_encoder 应该部分可训练!")
            print(f"   ⚠️ encoder_partial 实际等同于 decoder_prompt!")
        else:
            print(f"   ✅ vision_encoder 部分已解冻")

    elif strategy == "encoder_full":
        if module_stats.get('vision_encoder', {}).get('trainable', 0) == 0:
            print(f"   ❌ 错误: vision_encoder 应该全部可训练!")
        else:
            print(f"   ✅ vision_encoder 已全部解冻")

    elif strategy == "full":
        total_trainable = sum(s['trainable'] for s in module_stats.values())
        total_params = sum(s['total'] for s in module_stats.values())
        if total_trainable < total_params * 0.99:
            print(f"   ⚠️ 警告: full策略但不是所有参数都可训练")
        else:
            print(f"   ✅ 所有参数已解冻")
            
    elif strategy == "lora":
        # 检查是否有lora参数
        lora_params = sum(1 for n, p in sam.named_parameters() if 'lora' in n.lower() and p.requires_grad)
        if lora_params == 0:
            print(f"   ❌ 错误: 没有找到可训练的LoRA参数!")
        else:
            print(f"   ✅ 找到 {lora_params} 个LoRA参数")
    
    # 【新增】LoRA+ 策略验证
    elif strategy in ["lora_plus_encoder", "lora_plus_decoder"]:
        lora_params = sum(1 for n, p in sam.named_parameters() if 'lora' in n.lower() and p.requires_grad)
        if lora_params == 0:
            print(f"   ❌ 错误: 没有找到可训练的LoRA参数!")
        else:
            print(f"   ✅ 找到 {lora_params} 个LoRA参数")
        
        # 检查 decoder 是否被解冻（对于 lora_plus_encoder）
        if strategy == "lora_plus_encoder":
            if module_stats.get('mask_decoder', {}).get('trainable', 0) == 0:
                print(f"   ⚠️ 警告: mask_decoder 未被解冻")
            else:
                print(f"   ✅ mask_decoder 已解冻（联合训练）")
    
    return module_stats


# ========================== LoRA+ 实现 ==========================

def get_lora_target_modules_for_model(model_name: str, component: str = "encoder") -> List[str]:
    """
    根据模型类型返回正确的 LoRA target_modules

    Args:
        model_name: 模型名称 ('medsam' 或 'sammed2d')
        component: 'encoder' 或 'decoder'

    Returns:
        target_modules 列表

    注意 - SAM-Med2D qkv 层的特殊性:
        SAM-Med2D encoder 的 qkv 层是 nn.Linear(dim, dim * 3)，输出维度是输入的 3 倍（Q/K/V 合并）。
        对 qkv 层加 LoRA 会同时调整 Q、K、V 三个投影，无法像 MedSAM 那样只调 Q 和 V 而不调 K。
        这意味着:
        1. 参数量比分开的 q_proj/v_proj 约多 1.5 倍（因为也包含了 K 的调整）
        2. 语义上是对整个 attention 投影做 low-rank 调整，而非仅调 Q/V
    """
    if model_name.lower() == "sammed2d":
        # SAM-Med2D 使用 segment_anything 库
        # ViT encoder 的 Attention 使用合并的 qkv 层: self.qkv = nn.Linear(dim, dim * 3)
        # mask_decoder 的 Attention 使用: self.q_proj, self.k_proj, self.v_proj
        if component == "encoder":
            return ["qkv", "proj"]  # encoder 使用合并的 qkv
        else:
            return ["q_proj", "v_proj", "out_proj"]  # decoder 使用分开的
    else:
        # MedSAM (HuggingFace transformers.SamModel)
        # vision_encoder 和 mask_decoder 都使用分开的 q_proj, v_proj
        return ["q_proj", "v_proj"]


class LoRAPlusLinear(nn.Module):
    """
    LoRA+ Linear 层实现
    核心创新: A矩阵和B矩阵使用不同的学习率
    B矩阵学习率 = A矩阵学习率 * lr_ratio (论文推荐 lr_ratio=16)
    """
    def __init__(
        self, 
        original_linear: nn.Linear, 
        r: int = 8, 
        alpha: int = 16, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # 冻结原始权重
        original_linear.weight.requires_grad = False
        if original_linear.bias is not None:
            original_linear.bias.requires_grad = False
        
        # LoRA 矩阵 - 这两个会被分配不同的学习率
        # lora_A: (r, in_features) - 使用基础学习率
        # lora_B: (out_features, r) - 使用 lr * lr_ratio
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)  # B初始化为0，保证初始输出不变
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        result = self.original_linear(x)
        
        # LoRA 增量: (x @ A^T) @ B^T * scaling
        lora_out = self.lora_dropout(x)
        lora_out = lora_out @ self.lora_A.T  # (batch, seq, r)
        lora_out = lora_out @ self.lora_B.T  # (batch, seq, out)
        lora_out = lora_out * self.scaling
        
        return result + lora_out


def _get_submodule(model, target_name: str):
    """根据点分隔的名称获取子模块"""
    parts = target_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def _set_submodule(model, target_name: str, new_module: nn.Module):
    """设置子模块"""
    parts = target_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _setup_lora_plus(model, config: FinetuneConfig, component: str, model_name: str):
    """
    设置 LoRA+ 微调

    Args:
        model: 模型包装器
        config: 微调配置
        component: 'encoder' 或 'decoder'
        model_name: 模型名称，用于自动选择 target_modules
    """
    if hasattr(model, 'model'):
        sam = model.model
    else:
        sam = model

    # 【修复】根据模型类型自动选择 target_modules
    target_modules = get_lora_target_modules_for_model(model_name, component)

    print(f"\n{'='*70}")
    print(f"🔧 设置 LoRA+ ({component}) - 模型: {model_name}")
    print(f"   r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"   lr_ratio={config.lora_plus_lr_ratio} (B矩阵学习率倍数)")
    print(f"   target_modules={target_modules} (自动检测)")
    print(f"{'='*70}")
    
    # 确定搜索范围
    if component == "encoder":
        if hasattr(sam, 'vision_encoder'):
            search_root = sam.vision_encoder
            root_name = 'vision_encoder'
        elif hasattr(sam, 'image_encoder'):
            search_root = sam.image_encoder
            root_name = 'image_encoder'
        else:
            print(f"   ❌ 错误: 找不到 encoder 模块!")
            return
        search_roots = [(search_root, root_name)]
    elif component == "decoder":
        # 【修复】只在 mask_decoder 中搜索，prompt_encoder 基本没有 attention 层
        search_roots = []
        if hasattr(sam, 'mask_decoder'):
            search_roots.append((sam.mask_decoder, 'mask_decoder'))
        # 移除 prompt_encoder 搜索（它主要是 Embedding 层，不会有 q_proj/v_proj）
    else:
        raise ValueError(f"Unknown component: {component}")
    
    applied_count = 0
    
    for search_root, root_name in search_roots:
        for name, module in search_root.named_modules():
            if isinstance(module, nn.Linear):
                # 检查是否是目标模块
                full_name = f"{root_name}.{name}" if name else root_name
                is_target = False
                for target in target_modules:
                    if target in name or name.endswith(target):
                        is_target = True
                        break
                
                if is_target:
                    # 创建 LoRA+ 模块
                    lora_module = LoRAPlusLinear(
                        original_linear=module,
                        r=config.lora_r,
                        alpha=config.lora_alpha,
                        dropout=config.lora_dropout
                    )
                    
                    # 替换原模块
                    _set_submodule(search_root, name, lora_module)
                    
                    applied_count += 1
                    print(f"      ✅ {full_name}")
    
    if applied_count == 0:
        print(f"   ⚠️ 警告: 未找到任何目标模块!")
        print(f"   检查 target_modules: {target_modules}")
    else:
        print(f"\n   ✅ LoRA+ 已应用到 {applied_count} 个模块")
    
    # 统计 LoRA 参数
    lora_a_params = 0
    lora_b_params = 0
    for name, param in sam.named_parameters():
        if 'lora_A' in name and param.requires_grad:
            lora_a_params += param.numel()
        elif 'lora_B' in name and param.requires_grad:
            lora_b_params += param.numel()
    
    print(f"\n   📊 LoRA+ 参数统计:")
    print(f"      lora_A 参数: {lora_a_params:,}")
    print(f"      lora_B 参数: {lora_b_params:,}")
    print(f"      总 LoRA 参数: {lora_a_params + lora_b_params:,}")


# ========================== MedSAM 微调 ==========================

def setup_medsam_finetune(model, config: FinetuneConfig):
    """设置MedSAM微调 - 带完整诊断"""
    strategy = config.strategy
    sam = model.model  # transformers.SamModel
    
    print(f"\n{'='*70}")
    print(f"🔧 设置MedSAM微调")
    print(f"   策略: {strategy}")
    print(f"   unfreeze_encoder_layers: {config.unfreeze_encoder_layers}")
    print(f"{'='*70}")
    
    # 诊断模型结构
    diagnose_model_structure(model, "MedSAM")
    
    # Step 1: 冻结所有参数
    print(f"\n[Step 1] 冻结所有参数...")
    freeze_module(sam)
    t1, _ = count_parameters(sam)
    print(f"   冻结后可训练参数: {t1:,}")
    
    # Step 2: 根据策略解冻
    print(f"\n[Step 2] 根据策略 '{strategy}' 解冻参数...")
    
    if strategy == "decoder_only":
        unfreeze_module(sam.mask_decoder)
        print(f"   解冻: mask_decoder")
        
    elif strategy == "decoder_prompt":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        print(f"   解冻: mask_decoder, prompt_encoder")
        
    elif strategy == "encoder_partial":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        print(f"   解冻: mask_decoder, prompt_encoder")
        
        # 解冻encoder最后N层
        n_layers = config.unfreeze_encoder_layers
        encoder = sam.vision_encoder
        
        # 查找encoder的layers
        layers = None
        layer_attr = None
        for attr in ['layers', 'blocks', 'layer', 'encoder_layers']:
            if hasattr(encoder, attr):
                candidate = getattr(encoder, attr)
                if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                    layers = candidate
                    layer_attr = attr
                    break
        
        if layers is not None:
            total = len(layers)
            if n_layers > 0:
                start_idx = max(0, total - n_layers)
                for i in range(start_idx, total):
                    unfreeze_module(layers[i])
                    print(f"   解冻: encoder.{layer_attr}[{i}]")
                print(f"   ✅ 共解冻encoder {n_layers}/{total} 层")
            else:
                print(f"   ⚠️ n_layers=0, encoder不会被解冻!")
        else:
            print(f"   ❌ 错误: 找不到encoder的layers!")
            print(f"   ⚠️ encoder_partial 将等同于 decoder_prompt!")

    elif strategy == "encoder_full":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        unfreeze_module(sam.vision_encoder)
        print(f"   解冻: mask_decoder, prompt_encoder, vision_encoder (全部)")

    elif strategy == "full":
        unfreeze_module(sam)
        print(f"   解冻: 全部参数")
        
    elif strategy == "lora":
        print(f"   应用LoRA...")
        _setup_lora_medsam(model, config)

    # 【新增】LoRA+ encoder 策略
    elif strategy == "lora_plus_encoder":
        print(f"   应用 LoRA+ (encoder)...")
        _setup_lora_plus(model, config, component="encoder", model_name="medsam")
        # 同时解冻 decoder 进行常规训练
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        print(f"   ✅ 同时解冻 decoder 和 prompt_encoder 进行常规训练")

    # 【新增】LoRA+ decoder 策略
    elif strategy == "lora_plus_decoder":
        print(f"   应用 LoRA+ (decoder)...")
        _setup_lora_plus(model, config, component="decoder", model_name="medsam")
        # 【修复】也解冻 prompt_encoder，增加可训练参数
        unfreeze_module(sam.prompt_encoder)
        print(f"   ✅ 同时解冻 prompt_encoder 进行常规训练")

    else:
        raise ValueError(f"未知策略: {strategy}")

    # Step 3: 诊断结果
    print(f"\n[Step 3] 验证解冻结果...")
    diagnose_trainable_params(model, strategy)

    # 打印最终统计
    print_trainable_parameters(sam, "MedSAM")
    
    return model


def _setup_lora_medsam(model, config: FinetuneConfig):
    """为MedSAM设置LoRA"""
    sam = model.model
    
    # ✅ 检查是否已经是PEFT模型，避免重复包装
    try:
        from peft import PeftModel
        if isinstance(sam, PeftModel):
            print(f"   ⚠️ 模型已经是PEFT模型，跳过重复包装")
            return
    except ImportError:
        pass
    
    try:
        from peft import get_peft_model, LoraConfig
        
        print(f"   使用peft库设置LoRA...")
        print(f"   r={config.lora_r}, alpha={config.lora_alpha}")
        print(f"   target_modules={config.lora_target_modules}")
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
        )
        
        # 应用LoRA - 这会修改模型结构
        model.model = get_peft_model(sam, lora_config)
        print(f"   ✅ LoRA配置成功")
        
        # 打印LoRA参数
        lora_params = [(n, p.numel()) for n, p in model.model.named_parameters() 
                       if 'lora' in n.lower() and p.requires_grad]
        print(f"   LoRA参数数量: {len(lora_params)}")
        for name, count in lora_params[:5]:
            print(f"      - {name}: {count:,}")
        if len(lora_params) > 5:
            print(f"      ... 及其他 {len(lora_params)-5} 个")
            
    except ImportError as e:
        print(f"   ⚠️ peft库未安装: {e}")
        print(f"   使用手动LoRA实现...")
        _apply_manual_lora(sam, config)
    except Exception as e:
        print(f"   ❌ LoRA设置失败: {e}")
        raise


def _apply_manual_lora(model, config):
    """手动LoRA实现"""
    count = 0
    
    # ✅ 获取模型所在设备
    device = next(model.parameters()).device
    print(f"   模型设备: {device}")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for target in config.lora_target_modules:
                if target in name:
                    in_f, out_f = module.in_features, module.out_features
                    
                    # 冻结原始权重
                    module.weight.requires_grad = False
                    if module.bias is not None:
                        module.bias.requires_grad = False
                    
                    # ✅ 添加LoRA参数 - 在正确的设备上创建
                    lora_A = nn.Parameter(torch.zeros(config.lora_r, in_f, device=device))
                    lora_B = nn.Parameter(torch.zeros(out_f, config.lora_r, device=device))
                    nn.init.kaiming_uniform_(lora_A, a=5**0.5)
                    nn.init.zeros_(lora_B)
                    
                    module.register_parameter('lora_A', lora_A)
                    module.register_parameter('lora_B', lora_B)
                    module.scaling = config.lora_alpha / config.lora_r
                    
                    # 修改forward - ✅ 确保设备一致
                    original_forward = module.forward
                    def new_forward(x, orig_fwd=original_forward, mod=module):
                        result = orig_fwd(x)
                        # ✅ 确保LoRA参数在输入相同的设备上
                        lora_A = mod.lora_A.to(x.device)
                        lora_B = mod.lora_B.to(x.device)
                        lora_out = (x @ lora_A.T @ lora_B.T) * mod.scaling
                        return result + lora_out
                    module.forward = new_forward
                    
                    count += 1
                    print(f"      应用LoRA: {name}")
                    break
    
    print(f"   手动LoRA应用到 {count} 个层")


# ========================== SAM-Med2D 微调 ==========================

def setup_sammed2d_finetune(model, config: FinetuneConfig):
    """设置SAM-Med2D微调"""
    strategy = config.strategy
    sam = model.model
    
    print(f"\n{'='*70}")
    print(f"🔧 设置SAM-Med2D微调")
    print(f"   策略: {strategy}")
    print(f"{'='*70}")
    
    diagnose_model_structure(model, "SAM-Med2D")
    
    freeze_module(sam)
    
    if strategy == "decoder_only":
        unfreeze_module(sam.mask_decoder)
        
    elif strategy == "decoder_prompt":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        
    elif strategy == "adapter_only":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        
        adapter_count = 0
        for name, param in sam.image_encoder.named_parameters():
            if 'adapter' in name.lower() or 'Adapter' in name:
                param.requires_grad = True
                adapter_count += 1
        print(f"   解冻 {adapter_count} 个adapter参数")
        
    elif strategy == "encoder_partial":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        
        n_layers = config.unfreeze_encoder_layers
        if hasattr(sam.image_encoder, 'blocks') and n_layers > 0:
            total = len(sam.image_encoder.blocks)
            for i in range(total - n_layers, total):
                unfreeze_module(sam.image_encoder.blocks[i])
        
        for name, param in sam.image_encoder.named_parameters():
            if 'adapter' in name.lower():
                param.requires_grad = True

    elif strategy == "encoder_full":
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        unfreeze_module(sam.image_encoder)
        print(f"   解冻: mask_decoder, prompt_encoder, image_encoder (全部)")

    elif strategy == "full":
        unfreeze_module(sam)

    elif strategy == "lora":
        # 【修复】SAM-Med2D 需要使用不同的 target_modules
        config.lora_target_modules = get_lora_target_modules_for_model("sammed2d", "encoder")
        print(f"   SAM-Med2D LoRA target_modules: {config.lora_target_modules}")
        _setup_lora_medsam(model, config)

    # 【新增】LoRA+ encoder 策略
    elif strategy == "lora_plus_encoder":
        print(f"   应用 LoRA+ (encoder)...")
        _setup_lora_plus(model, config, component="encoder", model_name="sammed2d")
        # 同时解冻 decoder 进行常规训练
        unfreeze_module(sam.mask_decoder)
        unfreeze_module(sam.prompt_encoder)
        print(f"   ✅ 同时解冻 decoder 和 prompt_encoder 进行常规训练")

    # 【新增】LoRA+ decoder 策略
    elif strategy == "lora_plus_decoder":
        print(f"   应用 LoRA+ (decoder)...")
        _setup_lora_plus(model, config, component="decoder", model_name="sammed2d")
        # 【修复】也解冻 prompt_encoder，增加可训练参数
        unfreeze_module(sam.prompt_encoder)
        print(f"   ✅ 同时解冻 prompt_encoder 进行常规训练")

    diagnose_trainable_params(model, strategy)
    print_trainable_parameters(sam, "SAM-Med2D")
    
    return model


# ========================== 入口函数 ==========================

def setup_finetune(model, model_name: str, config: FinetuneConfig):
    """统一入口"""
    print(f"\n{'#'*70}")
    print(f"# 微调设置开始")
    print(f"# 模型: {model_name}")
    print(f"# 策略: {config.strategy}")
    print(f"# 学习率: {config.learning_rate}")
    print(f"# warmup_epochs: {config.warmup_epochs}")
    print(f"# unfreeze_encoder_layers: {config.unfreeze_encoder_layers}")
    # 【新增】打印 LoRA+ 参数
    if config.strategy.startswith("lora"):
        print(f"# LoRA r: {config.lora_r}")
        print(f"# LoRA alpha: {config.lora_alpha}")
    if config.strategy.startswith("lora_plus"):
        print(f"# LoRA+ lr_ratio: {config.lora_plus_lr_ratio}")
    print(f"{'#'*70}")
    
    if model_name.lower() == "medsam":
        result = setup_medsam_finetune(model, config)
    elif model_name.lower() == "sammed2d":
        result = setup_sammed2d_finetune(model, config)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    print(f"\n{'#'*70}")
    print(f"# 微调设置完成")
    print(f"{'#'*70}\n")
    
    return result


# ========================== 优化器 ==========================

def build_optimizer(model, config: FinetuneConfig) -> torch.optim.Optimizer:
    """
    构建优化器
    
    【新增】对于 LoRA+ 策略，A矩阵和B矩阵使用不同的学习率:
    - lora_A: 基础学习率 lr
    - lora_B: lr * lora_plus_lr_ratio (论文推荐 16x)
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    print(f"\n📊 构建优化器:")
    print(f"   可训练参数tensor数: {len(trainable_params)}")
    
    if len(trainable_params) == 0:
        print(f"   ❌ 错误: 没有可训练参数!")
        raise RuntimeError("没有可训练参数! 检查freeze/unfreeze逻辑")
    
    total_params = sum(p.numel() for p in trainable_params)
    print(f"   可训练参数总量: {total_params:,}")
    
    # 【新增】LoRA+ 模式: 分离 A 和 B 矩阵的参数组
    if config.strategy.startswith("lora_plus"):
        lora_a_params = []
        lora_b_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'lora_A' in name:
                lora_a_params.append(param)
            elif 'lora_B' in name:
                lora_b_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        
        if lora_a_params:
            param_groups.append({
                "params": lora_a_params,
                "lr": config.learning_rate,
                "name": "lora_A"
            })
            print(f"   lora_A组: {len(lora_a_params)} tensors, lr={config.learning_rate}")
        
        if lora_b_params:
            lora_b_lr = config.learning_rate * config.lora_plus_lr_ratio
            param_groups.append({
                "params": lora_b_params,
                "lr": lora_b_lr,
                "name": "lora_B"
            })
            print(f"   lora_B组: {len(lora_b_params)} tensors, lr={lora_b_lr} ({config.lora_plus_lr_ratio}x)")
        
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": config.learning_rate,
                "name": "other"
            })
            print(f"   other组: {len(other_params)} tensors, lr={config.learning_rate}")
        
        if len(param_groups) == 0:
            raise RuntimeError("没有可训练参数! 检查freeze/unfreeze逻辑")
    
    # 分层学习率 (原有逻辑)
    elif config.strategy in ["encoder_partial", "encoder_full", "full"]:
        encoder_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "encoder" in name.lower():
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": config.learning_rate * config.encoder_lr_scale,
                "name": "encoder"
            })
            print(f"   encoder组: {len(encoder_params)} tensors, lr={config.learning_rate * config.encoder_lr_scale}")
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": config.learning_rate,
                "name": "other"
            })
            print(f"   other组: {len(other_params)} tensors, lr={config.learning_rate}")
    else:
        param_groups = [{"params": trainable_params, "lr": config.learning_rate}]
        print(f"   统一学习率: {config.learning_rate}")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    return optimizer


def build_scheduler(optimizer, config: FinetuneConfig, steps_per_epoch: int):
    """构建学习率调度器"""
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = int(config.warmup_epochs * steps_per_epoch)  # 【修复】支持小数warmup_epochs
    
    print(f"\n📊 构建学习率调度器:")
    print(f"   总步数: {total_steps}")
    print(f"   Warmup步数: {warmup_steps} ({config.warmup_epochs} epochs)")
    
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)  # 【修复】从step+1开始，避免lr=0
        else:
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


# ========================== 损失函数 ==========================

class SegmentationLoss(nn.Module):
    """BCE + Dice损失 - 【修复】支持AMP混合精度训练"""
    def __init__(self, bce_weight: float = 0.3, dice_weight: float = 0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 【修复】在计算BCE前退出autocast，确保数值稳定性
        with torch.amp.autocast('cuda', enabled=False):
            # 转换为float32以确保精度
            pred = pred.float()
            target = target.float()
            
            # 确保预测值在有效范围内，避免log(0)
            pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
            
            # BCE Loss
            bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')
            
            # Dice Loss
            smooth = 1e-5
            pred_flat = pred.view(-1)
            target_flat = target.view(-1)
            intersection = (pred_flat * target_flat).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
            
            total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss


# ========================== 模型保存/加载 ==========================

def save_finetuned_model(model, config: FinetuneConfig, epoch: int, optimizer=None, metrics=None):
    """保存微调模型"""
    os.makedirs(config.output_dir, exist_ok=True)
    save_path = os.path.join(config.output_dir, f"checkpoint_epoch_{epoch}.pth")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics
        
    torch.save(checkpoint, save_path)
    print(f"✅ 模型已保存: {save_path}")
    return save_path


def load_finetuned_model(model, checkpoint_path: str, strict: bool = True):
    """加载微调模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    print(f"✅ 已加载检查点: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    if "metrics" in checkpoint:
        print(f"   Metrics: {checkpoint['metrics']}")
    
    return model, checkpoint


# ========================== 策略推荐 ==========================

def recommend_strategy(dataset_size: int, gpu_memory_gb: float, target_domain: str = "medical") -> str:
    print(f"\n📊 分析微调条件:")
    print(f"   数据集大小: {dataset_size}")
    print(f"   GPU显存: {gpu_memory_gb} GB")
    
    if dataset_size < 100:
        strategy = "decoder_only"
    elif dataset_size < 500:
        strategy = "decoder_prompt"
    elif dataset_size < 2000:
        strategy = "encoder_partial"
    else:
        strategy = "full" if gpu_memory_gb >= 24 else "encoder_partial"
    
    print(f"   推荐策略: {strategy}")
    return strategy
