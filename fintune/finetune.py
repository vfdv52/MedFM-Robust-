#!/usr/bin/env python3
"""
微调训练脚本 - 支持MedSAM和SAM-Med2D的微调
基于现有pipeline最小化修改

使用示例:
=========
# 1. Decoder-only微调 (最快，推荐小数据集)
python finetune.py \
    --model_name medsam \
    --strategy decoder_only \
    --data_path /path/to/data \
    --num_epochs 10

# 2. Adapter微调 (SAM-Med2D专用)
python finetune.py \
    --model_name sammed2d \
    --strategy adapter_only \
    --data_path /path/to/data \
    --num_epochs 20

# 3. 部分Encoder微调
python finetune.py \
    --model_name medsam \
    --strategy encoder_partial \
    --unfreeze_encoder_layers 4 \
    --data_path /path/to/data

# 4. LoRA微调 (参数高效)
python finetune.py \
    --model_name medsam \
    --strategy lora \
    --lora_r 8 \
    --data_path /path/to/data
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# 导入现有模块
from model_zoo import create_model, list_available_models
from finetune_utils import (
    FinetuneConfig, 
    setup_finetune, 
    build_optimizer, 
    build_scheduler,
    SegmentationLoss,
    save_finetuned_model,
    load_finetuned_model,
    recommend_strategy,
    print_trainable_parameters
)


# ========================== 数据集类 ==========================

class SegmentationDataset(Dataset):
    """
    通用医学图像分割数据集
    支持的数据格式:
    - 图像: jpg, png, npy
    - 掩码: png (二值)
    - bbox: json文件
    """
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        bbox_json: str,
        image_size: int = 256,
        is_train: bool = True,
        augmentation: bool = True,
        model_name: str = "medsam"
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.is_train = is_train
        self.augmentation = augmentation and is_train
        self.model_name = model_name
        
        # 加载bbox
        self.bbox_dict = {}
        if os.path.exists(bbox_json):
            with open(bbox_json, 'r') as f:
                self.bbox_dict = json.load(f)
        
        # 收集有效的图像-掩码对
        self.samples = self._collect_samples()
        print(f"✅ 数据集加载: {len(self.samples)} 个样本 ({'训练' if is_train else '验证'})")
        
    def _collect_samples(self) -> List[Dict]:
        """收集有效的图像-掩码对"""
        samples = []
        extensions = ['.jpg', '.jpeg', '.png', '.npy']
        
        for img_name in os.listdir(self.img_dir):
            if not any(img_name.lower().endswith(ext) for ext in extensions):
                continue
                
            base_name = os.path.splitext(img_name)[0]
            
            # 查找对应的掩码
            mask_path = None
            # for mask_suffix in ['_Segmentation.png', '_mask.png', '.png']:
            for mask_suffix in ['_Segmentation.png', '_mask.png', '.png', '.jpg', '.jpeg']:
                potential_mask = os.path.join(self.mask_dir, base_name + mask_suffix)
                if os.path.exists(potential_mask):
                    mask_path = potential_mask
                    break
            
            if mask_path is None:
                continue
                
            # 获取bbox
            mask_name = os.path.basename(mask_path)
            bbox = self.bbox_dict.get(mask_name, None)
            
            samples.append({
                'img_path': os.path.join(self.img_dir, img_name),
                'mask_path': mask_path,
                'bbox': bbox,
                'base_name': base_name
            })
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_image(self, path: str) -> np.ndarray:
        """加载图像"""
        if path.endswith('.npy'):
            img = np.load(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.uint8)
    
    def _load_mask(self, path: str) -> np.ndarray:
        """加载掩码"""
        mask = cv2.imread(path, 0)
        mask = (mask > 0).astype(np.float32)
        return mask
    
    def _resize(self, img: np.ndarray, mask: np.ndarray, bbox: list) -> Tuple:
        """调整尺寸"""
        orig_h, orig_w = img.shape[:2]
        target_size = (self.image_size, self.image_size)
        
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # 缩放bbox
        if bbox is not None:
            scale_w = self.image_size / orig_w
            scale_h = self.image_size / orig_h
            bbox = [
                bbox[0] * scale_w,
                bbox[1] * scale_h,
                bbox[2] * scale_w,
                bbox[3] * scale_h
            ]
            
        return img, mask, bbox
    
    def _augment(self, img: np.ndarray, mask: np.ndarray, bbox: list) -> Tuple:
        """数据增强"""
        if not self.augmentation:
            return img, mask, bbox
            
        h, w = img.shape[:2]
        
        # 随机水平翻转
        if np.random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            if bbox is not None:
                bbox = [w - bbox[2], bbox[1], w - bbox[0], bbox[3]]
        
        # 随机垂直翻转
        if np.random.random() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
            if bbox is not None:
                bbox = [bbox[0], h - bbox[3], bbox[2], h - bbox[1]]
        
        # 随机亮度/对比度调整
        if np.random.random() < 0.3:
            alpha = 0.8 + np.random.random() * 0.4  # 0.8-1.2
            beta = -20 + np.random.random() * 40    # -20 to 20
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            
        return img, mask, bbox
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 加载数据
        img = self._load_image(sample['img_path'])
        mask = self._load_mask(sample['mask_path'])
        bbox = sample['bbox'].copy() if sample['bbox'] else None
        
        # 调整尺寸
        img, mask, bbox = self._resize(img, mask, bbox)
        
        # 数据增强
        img, mask, bbox = self._augment(img, mask, bbox)
        
        # 转换为tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        
        result = {
            'image': img_tensor,
            'mask': mask_tensor,
            'name': sample['base_name']
        }
        
        if bbox is not None:
            result['bbox'] = torch.tensor(bbox, dtype=torch.float32)
            
        return result


def collate_fn(batch: List[Dict]) -> Dict:
    """自定义批处理函数"""
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    names = [item['name'] for item in batch]
    
    result = {
        'image': images,
        'mask': masks,
        'name': names
    }
    
    if 'bbox' in batch[0]:
        # bbox不能直接stack，保持为list
        result['bbox'] = [item.get('bbox') for item in batch]
        
    return result


# ========================== 训练器 ==========================

class Trainer:
    """微调训练器"""
    
    def __init__(
        self,
        model,
        model_name: str,
        config: FinetuneConfig,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 设置微调
        self.model = setup_finetune(model, model_name, config)
        self.model.to(device)
        
        # 构建优化器和调度器
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(
            self.optimizer, 
            config, 
            len(train_loader) // config.gradient_accumulation_steps
        )
        
        # 损失函数 - 【修复】调整权重，Dice占更大比重
        self.criterion = SegmentationLoss(bce_weight=0.3, dice_weight=0.7)
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # 训练记录
        self.best_dice = 0.0
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'test_dice': [],
            'test_iou': []
        }
        
    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            bboxes = batch.get('bbox', [None] * len(images))
            
            # 【调试】第一个batch打印数据分布
            if batch_idx == 0 and epoch == 0:
                print(f"\n[DEBUG] 数据分布检查:")
                print(f"  image: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
                print(f"  mask: min={masks.min():.4f}, max={masks.max():.4f}, mean={masks.mean():.4f}")
                print(f"  mask前景比例: {(masks > 0.5).float().mean():.4f}")
            
            # 前向传播
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                batch_loss = 0.0
                
                for i in range(len(images)):
                    img = images[i:i+1]
                    mask = masks[i:i+1]
                    bbox = bboxes[i].tolist() if bboxes[i] is not None else None
                    
                    # 模型前向
                    if self.model.prompt_required and bbox is not None:
                        pred_mask, _ = self.model(img, bbox, mask)
                    else:
                        pred_mask, _ = self.model(img, None, mask)
                    
                    # 【修复】始终使用criterion计算loss（BCE+Dice），忽略模型返回的loss
                    loss = self.criterion(pred_mask, mask)
                    
                    batch_loss += loss
                    
                    # 【调试】第一个样本打印预测分布
                    if batch_idx == 0 and epoch == 0 and i == 0:
                        print(f"  pred: min={pred_mask.min():.4f}, max={pred_mask.max():.4f}, mean={pred_mask.mean():.4f}")
                        print(f"  loss: {loss.item():.4f}")
                    
                batch_loss = batch_loss / len(images)
                batch_loss = batch_loss / self.config.gradient_accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()
            
            # 梯度更新
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            total_loss += batch_loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """测试"""
        if self.test_loader is None:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        num_samples = 0
        
        for batch in tqdm(self.test_loader, desc="Testing"):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            bboxes = batch.get('bbox', [None] * len(images))
            
            for i in range(len(images)):
                img = images[i:i+1]
                mask = masks[i:i+1]
                bbox = bboxes[i].tolist() if bboxes[i] is not None else None
                
                # 前向传播
                if self.model.prompt_required and bbox is not None:
                    pred_mask = self.model(img, bbox)
                else:
                    pred_mask = self.model(img, None)
                
                if isinstance(pred_mask, tuple):
                    pred_mask = pred_mask[0]
                
                # 计算损失
                loss = self.criterion(pred_mask, mask)
                total_loss += loss.item()
                
                # 计算指标
                pred_bin = (pred_mask > 0.5).float()
                mask_bin = (mask > 0.5).float()
                
                intersection = (pred_bin * mask_bin).sum()
                union = pred_bin.sum() + mask_bin.sum() - intersection
                
                iou = (intersection / (union + 1e-6)).item()
                dice = (2 * intersection / (pred_bin.sum() + mask_bin.sum() + 1e-6)).item()
                
                total_iou += iou
                total_dice += dice
                num_samples += 1
                
        return {
            'loss': total_loss / num_samples,
            'dice': total_dice / num_samples,
            'iou': total_iou / num_samples
        }
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"🚀 开始训练")
        print(f"   模型: {self.model_name}")
        print(f"   策略: {self.config.strategy}")
        print(f"   设备: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # 测试
            if self.test_loader and (epoch + 1) % self.config.eval_interval == 0:
                test_metrics = self.validate()
                self.history['test_loss'].append(test_metrics.get('loss', 0))
                self.history['test_dice'].append(test_metrics.get('dice', 0))
                self.history['test_iou'].append(test_metrics.get('iou', 0))
                
                print(f"\n📊 Epoch {epoch+1} 结果:")
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Test Loss: {test_metrics['loss']:.4f}")
                print(f"   Test Dice: {test_metrics['dice']:.4f}")
                print(f"   Test IoU: {test_metrics['iou']:.4f}")
                
                # 保存最佳模型
                if test_metrics['dice'] > self.best_dice:
                    self.best_dice = test_metrics['dice']
                    # 保存为 best_model.pth
                    best_path = os.path.join(self.config.output_dir, "best_model.pth")
                    checkpoint = {
                        "epoch": epoch + 1,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config.to_dict(),
                        "metrics": test_metrics
                    }
                    torch.save(checkpoint, best_path)
                    print(f"   ✅ 新的最佳模型! Dice: {self.best_dice:.4f}")
                    print(f"   💾 已保存: {best_path}")
            else:
                print(f"\n📊 Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
            
            # 注释掉定期保存，只保留best_model.pth节省空间
            # if (epoch + 1) % self.config.save_interval == 0:
            #     save_finetuned_model(
            #         self.model,
            #         self.config,
            #         epoch + 1,
            #         self.optimizer
            #     )
                
        print(f"\n{'='*60}")
        print(f"✅ 训练完成!")
        print(f"   最佳Dice: {self.best_dice:.4f}")
        print(f"{'='*60}\n")
        
        return self.history


# ========================== 参数解析 ==========================

def get_parser():
    parser = argparse.ArgumentParser(description="医学图像分割模型微调")
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default="medsam",
                        choices=list_available_models(),
                        help="模型名称")
    parser.add_argument("--model_config", type=str, default="model_config.json",
                        help="模型配置文件")
    
    # 数据配置
    parser.add_argument("--data_path", type=str, required=True,
                        help="数据集路径 (包含images, masks, bbox.json)")
    parser.add_argument("--img_dir", type=str, default=None,
                        help="图像目录 (默认: data_path/images)")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="掩码目录 (默认: data_path/masks)")
    parser.add_argument("--bbox_json", type=str, default=None,
                        help="bbox文件 (默认: data_path/bbox_coordinates.json)")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="测试集比例")
    
    # 微调策略
    parser.add_argument("--strategy", type=str, default="decoder_only",
                        choices=["decoder_only", "decoder_prompt", "adapter_only",
                                "encoder_partial", "encoder_full", "full", "lora",
                                "lora_plus_encoder", "lora_plus_decoder"],
                        help="微调策略")
    parser.add_argument("--unfreeze_encoder_layers", type=int, default=2,
                        help="解冻encoder最后N层 (encoder_partial策略)")
    
    # LoRA配置
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    
    # 【新增】LoRA+ 配置
    parser.add_argument("--lora_plus_lr_ratio", type=float, default=16.0,
                        help="LoRA+ B矩阵学习率倍数 (论文推荐16)")
    
    # 训练配置
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")  # 【修复】从1e-4提高到5e-4
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./finetune_output",
                        help="输出目录")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    
    # 其他
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--recommend", action="store_true",
                        help="推荐微调策略")
    
    return parser


# ========================== 主函数 ==========================

def main():
    args = get_parser().parse_args()
    
    # 设置路径
    data_path = Path(args.data_path)
    img_dir = args.img_dir or str(data_path / "images")
    mask_dir = args.mask_dir or str(data_path / "masks") 
    bbox_json = args.bbox_json or str(data_path / "bbox_coordinates.json")
    
    # 策略推荐
    if args.recommend:
        # 统计数据量
        num_images = len(list(Path(img_dir).glob("*")))
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        
        strategy = recommend_strategy(
            dataset_size=num_images,
            gpu_memory_gb=gpu_memory
        )
        print(f"\n使用推荐策略: {strategy}")
        args.strategy = strategy
    
    # 创建配置
    config = FinetuneConfig(
        strategy=args.strategy,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        unfreeze_encoder_layers=args.unfreeze_encoder_layers,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_plus_lr_ratio=args.lora_plus_lr_ratio,  # 【新增】LoRA+ 参数
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        output_dir=args.output_dir
    )
    
    # 加载模型配置
    model_cfg = {}
    if os.path.exists(args.model_config):
        with open(args.model_config, 'r') as f:
            all_cfg = json.load(f)
            model_cfg = all_cfg.get(args.model_name, {})
    
    # 创建模型
    print(f"\n📦 加载模型: {args.model_name}")
    model = create_model(args.model_name, args.device, model_cfg)
    
    # 创建数据集
    print(f"\n📂 加载数据集: {args.data_path}")
    
    full_dataset = SegmentationDataset(
        img_dir=img_dir,
        mask_dir=mask_dir,
        bbox_json=bbox_json,
        image_size=args.image_size,
        is_train=True,
        augmentation=True,
        model_name=args.model_name
    )
    
    # 划分训练/测试集
    test_size = int(len(full_dataset) * args.test_split)
    train_size = len(full_dataset) - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 保存测试集文件名列表（供 pipeline.py 使用）
    os.makedirs(config.output_dir, exist_ok=True)
    test_indices = test_dataset.indices
    test_samples_info = {
        "test_files": [full_dataset.samples[i]['base_name'] for i in test_indices],
        "train_files": [full_dataset.samples[i]['base_name'] for i in train_dataset.indices],
        "test_split": args.test_split,
        "random_seed": 42,
        "total_samples": len(full_dataset),
        "test_samples": test_size,
        "train_samples": train_size
    }
    test_samples_path = os.path.join(config.output_dir, "data_split.json")
    with open(test_samples_path, 'w') as f:
        json.dump(test_samples_info, f, indent=2)
    print(f"   💾 数据划分已保存: {test_samples_path}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"   训练样本: {train_size}")
    print(f"   测试样本: {test_size}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        model_name=args.model_name,
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        device=args.device
    )
    
    # 恢复训练
    if args.resume:
        model, checkpoint = load_finetuned_model(model, args.resume)
        print(f"✅ 从检查点恢复: {args.resume}")
    
    # 开始训练
    history = trainer.train()
    
    # 保存训练历史
    history_path = os.path.join(config.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"✅ 训练历史已保存: {history_path}")


if __name__ == "__main__":
    main()
