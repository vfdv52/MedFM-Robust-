# model_zoo.py
import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List
from segment_anything import sam_model_registry

# ---------------------------- 基类 ----------------------------
class SegmentationModelBase(nn.Module, ABC):
    def __init__(self, device, cfg):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.name = cfg.get("name", "Unknown")
        self.prompt_type = cfg.get("prompt_type", "none")
        self.prompt_required = cfg.get("prompt_required", False)
        
    @abstractmethod
    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        """
        Args:
            img_tensor: [B, C, H, W]
            input_boxes: 提示信息 (box/point/none)
            target_mask: [B, 1, H, W] 用于训练时计算loss
        Returns:
            pred_masks: [B, 1, H, W]
            loss: Optional[Tensor] 如果提供target_mask
        """
        pass

    def predict(self, img_tensor, input_boxes=None):
        """推理模式，只返回预测结果"""
        with torch.no_grad():
            return self.forward(img_tensor, input_boxes)


# ---------------------------- SAM系列模型 ----------------------------
class MedSAMModel(SegmentationModelBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        try:
            from transformers import SamModel, SamProcessor
            self.model = SamModel.from_pretrained(cfg["repo_id"]).to(device)
            self.processor = SamProcessor.from_pretrained(cfg["repo_id"])
            self.processor.image_processor.do_rescale = False
            print(f"✅ 成功加载 {cfg['name']}")
        except Exception as e:
            print(f"❌ 加载 {cfg['name']} 失败: {e}")
            raise

    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        img_tensor = img_tensor.to(self.device)
        
        # 检查是否需要梯度
        if img_tensor.requires_grad:
            # 对抗攻击模式，确保梯度保持
            inputs = self.processor(
                images=img_tensor,
                input_boxes=[[input_boxes]] if input_boxes is not None else None,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # 如果processor断开了梯度，尝试修复
            if not inputs['pixel_values'].requires_grad:
                processed_shape = inputs['pixel_values'].shape
                other_inputs = {k: v for k, v in inputs.items() if k != 'pixel_values'}
                
                # 手动处理图像保持梯度
                if processed_shape[-2:] != img_tensor.shape[-2:]:
                    processed_img = F.interpolate(
                        img_tensor,
                        size=processed_shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                else:
                    processed_img = img_tensor
                
                inputs = {**other_inputs, 'pixel_values': processed_img}
        else:
            # 推理模式，正常使用processor
            inputs = self.processor(
                images=img_tensor,
                input_boxes=[[input_boxes]] if input_boxes is not None else None,
                return_tensors="pt",
                padding=True
            ).to(self.device)
        
        # 模型前向传播
        outputs = self.model(**inputs, multimask_output=False)
        pred_masks = outputs.pred_masks.sigmoid()
        
        # 处理维度
        if len(pred_masks.shape) == 5:
            pred_masks = pred_masks.squeeze(2)
        
        # 调整尺寸到原始图像大小
        pred_masks = F.interpolate(
            pred_masks,
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        # 计算损失
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            if target_mask.shape != pred_masks.shape:
                target_mask = F.interpolate(
                    target_mask,
                    size=pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            with torch.amp.autocast('cuda', enabled=False):
            	loss = F.binary_cross_entropy(pred_masks.float(), target_mask.float())
            # loss = F.binary_cross_entropy(pred_masks, target_mask)
            return pred_masks, loss
        
        return pred_masks


class SAMModel(MedSAMModel):
    """与 MedSAM 完全一致，仅换权重"""
    pass


class SAMPointModel(SegmentationModelBase):
    """SAM 使用单点提示"""
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        try:
            from transformers import SamModel, SamProcessor
            self.model = SamModel.from_pretrained(cfg["repo_id"]).to(device)
            self.processor = SamProcessor.from_pretrained(cfg["repo_id"])
            self.processor.image_processor.do_rescale = False
            print(f"✅ 成功加载 {cfg['name']}")
        except Exception as e:
            print(f"❌ 加载 {cfg['name']} 失败: {e}")
            raise

    def forward(self, img_tensor, input_points=None, target_mask=None):
        img_tensor = img_tensor.to(self.device)
        
        inputs = self.processor(
            images=img_tensor,
            input_points=[[input_points]] if input_points is not None else None,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        outputs = self.model(**inputs, multimask_output=False)
        pred_masks = outputs.pred_masks.sigmoid()
        
        if len(pred_masks.shape) == 5:
            pred_masks = pred_masks.squeeze(2)
            
        pred_masks = F.interpolate(
            pred_masks,
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            # ✅ 使用autocast安全的方式计算loss
            with torch.amp.autocast('cuda', enabled=False):
                loss = F.binary_cross_entropy(pred_masks.float(), target_mask.float())
            return pred_masks, loss
            
        return pred_masks


# ---------------------------- 传统分割模型 ----------------------------
class UNetModel(SegmentationModelBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        try:
            # 使用简单的UNet实现或者预训练模型
            import torchvision.models.segmentation as seg_models
            self.model = seg_models.fcn_resnet50(pretrained=True, num_classes=1).to(device)
            
            # 如果有本地权重，加载
            if cfg.get("local_path") and os.path.exists(cfg["local_path"]):
                self.model.load_state_dict(torch.load(cfg["local_path"], map_location=device))
                print(f"✅ 从 {cfg['local_path']} 加载权重")
            else:
                print(f"✅ 使用预训练权重加载 {cfg['name']}")
        except Exception as e:
            print(f"❌ 加载 {cfg['name']} 失败: {e}")
            raise

    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        img_tensor = img_tensor.to(self.device)
        
        # UNet不需要提示，忽略input_boxes
        out = self.model(img_tensor)['out']
        pred_masks = torch.sigmoid(out)
        
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            if target_mask.shape != pred_masks.shape:
                target_mask = F.interpolate(
                    target_mask,
                    size=pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            # ✅ 使用autocast安全的方式计算loss
            with torch.amp.autocast('cuda', enabled=False):
                loss = F.binary_cross_entropy(pred_masks.float(), target_mask.float())
            return pred_masks, loss
            
        return pred_masks


class DeepLabV3Model(SegmentationModelBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        try:
            import torchvision.models.segmentation as seg_models
            self.model = seg_models.deeplabv3_resnet101(pretrained=True, num_classes=1).to(device)
            
            if cfg.get("local_path") and os.path.exists(cfg["local_path"]):
                self.model.load_state_dict(torch.load(cfg["local_path"], map_location=device))
                print(f"✅ 从 {cfg['local_path']} 加载权重")
            else:
                print(f"✅ 使用预训练权重加载 {cfg['name']}")
        except Exception as e:
            print(f"❌ 加载 {cfg['name']} 失败: {e}")
            raise

    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        img_tensor = img_tensor.to(self.device)
        
        out = self.model(img_tensor)['out']
        pred_masks = torch.sigmoid(out)
        
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
            if target_mask.shape != pred_masks.shape:
                target_mask = F.interpolate(
                    target_mask,
                    size=pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            loss = F.binary_cross_entropy(pred_masks, target_mask)
            return pred_masks, loss
            
        return pred_masks

# ---------------------------- SAM-Med2D ----------------------------

def create_official_args(model_type, ckpt_path, use_adapter, device):
    """完全复制官方的参数创建方式"""
    import argparse
    
    # 完全复制官方的 parse_args 函数
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="sammed", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument("--data_path", type=str, default="data_demo", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default=model_type, help="sam model_type")
    parser.add_argument("--sam_checkpoint", type=str, default=ckpt_path, help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=1, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num") 
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=use_adapter, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=False, help="save reslut")
    
    # 解析空的命令行参数，使用所有默认值
    args = parser.parse_args([])
    
    # 应用官方的后处理逻辑
    if args.iter_point > 1:
        args.iter_point = 1
        
    return args
  
# class SAMMed2DModel(SegmentationModelBase):
#     def __init__(self, device, cfg):
#         super().__init__(device, cfg)
#         from segment_anything import sam_model_registry
#         import os
    
#         model_type   = cfg.get("model_type", "vit_b")
#         ckpt_path    = cfg.get("local_path", "./pretrain_model/sam-med2d_b_clean.pth")
#         use_adapter  = cfg.get("encoder_adapter", True)
    
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
    
#         try:
#             # 使用与官方完全相同的参数创建方式
#             args = create_official_args(model_type, ckpt_path, use_adapter, device)
            
#             print(f"创建的 args 参数:")
#             for key, value in vars(args).items():
#                 print(f"  {key}: {value}")
                
#             # 使用与官方完全相同的模型创建方式
#             self.model = sam_model_registry[args.model_type](args).to(args.device)
#             self.model.eval()
#             print(f"✅ 成功加载 SAM-Med2D: {model_type} from {ckpt_path}")
            
#         except Exception as e:
#             print(f"❌ 加载 SAM-Med2D 失败: {e}")
#             print("请检查:")
#             print("1. segment_anything 库版本是否与官方一致")
#             print("2. 权重文件是否正确")
#             print("3. 是否使用了修改过的 SAM-Med2D 实现")
#             raise e

#     def forward(self, img_tensor, input_boxes=None, target_mask=None):
#         """
#         Args:
#             img_tensor: [B, C, H, W] in [0, 1]
#             input_boxes: list of [x1, y1, x2, y2] (原始坐标)
#             target_mask: [B, 1, H, W] 用于计算 loss
#         Returns:
#             pred_masks: [B, 1, H, W]
#             loss: 如果提供了 target_mask
#         """
        
#         from segment_anything.utils.transforms import ResizeLongestSide
#         transform = ResizeLongestSide(self.model.image_encoder.img_size)
    
#         # 转换为 numpy 图像
#         img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#         original_size = img_np.shape[:2]

#         # print("original_size:", original_size)          # 应该是 (256, 256)
    
#         # 官方预处理
        
#         input_image = transform.apply_image(img_np)
#         input_image_torch = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

#         # print("input_image_torch.shape:", input_image_torch.shape)  # 应该是 [1,3,H',W']，H'、W' ≤ 1024
    
#         # 图像编码
#         with torch.no_grad():
#             image_embedding = self.model.image_encoder(input_image_torch)
    		
#         if input_boxes is not None:
#             # 添加 bbox 坐标变换
#             box_np = np.array([input_boxes])
#             box_tf = transform.apply_boxes(box_np, original_size)
#             box = torch.tensor(box_tf, device=self.device).unsqueeze(0).float()
            
#             print(f"原始 bbox: {input_boxes}")
#             print(f"变换后 bbox: {box_tf[0]}")
#         else:
#             box = None

#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=None,
#             boxes=box,
#             masks=None,
#         )
    
#         # Mask 解码
#         low_res_masks, iou_predictions = self.model.mask_decoder(
#             image_embeddings=image_embedding,
#             image_pe=self.model.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )
    
#         # 后处理：还原到原图尺寸
#         pred_masks = torch.sigmoid(low_res_masks)
#         pred_masks = F.interpolate(
#             pred_masks,
#             size=original_size,
#             mode="bilinear",
#             align_corners=False
#         )
    
#         # 如果提供了 target_mask，计算 loss
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
#             if target_mask.shape != pred_masks.shape:
#                 target_mask = F.interpolate(target_mask, size=pred_masks.shape[-2:], mode="bilinear", align_corners=False)
#             loss = F.binary_cross_entropy(pred_masks, target_mask)
#             return pred_masks, loss
    
#         return pred_masks

# class SAMMed2DModel(SegmentationModelBase):
#     def __init__(self, device, cfg):
#         super().__init__(device, cfg)
#         from segment_anything import sam_model_registry
#         import os
    
#         model_type   = cfg.get("model_type", "vit_b")
#         ckpt_path    = cfg.get("local_path", "./pretrain_model/sam-med2d_b_clean.pth")
#         use_adapter  = cfg.get("encoder_adapter", True)
    
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
    
#         try:
#             # 🔑 直接使用配置中的image_size创建args
#             args = create_official_args(model_type, ckpt_path, use_adapter, device)
#             # 使用配置文件中的image_size (现在是256)
#             args.image_size = cfg.get("image_size", 256)
            
#             print(f"✅ SAM-Med2D 使用图像尺寸: {args.image_size}")
                
#             self.model = sam_model_registry[args.model_type](args).to(args.device)
#             self.model.eval()
#             print(f"✅ 成功加载 SAM-Med2D: {model_type} from {ckpt_path}")
            
#         except Exception as e:
#             print(f"❌ 加载 SAM-Med2D 失败: {e}")
#             raise e

#     def forward(self, img_tensor, input_boxes=None, target_mask=None):
#         """
#         Args:
#             img_tensor: [B, C, H, W] in [0, 1]
#             input_boxes: list of [x1, y1, x2, y2] (原始坐标)
#             target_mask: [B, 1, H, W] 用于计算 loss
#         Returns:
#             pred_masks: [B, 1, H, W]
#             loss: 如果提供了 target_mask
#         """
        
#         from segment_anything.utils.transforms import ResizeLongestSide
        
#         # 使用模型的image_encoder.img_size (现在应该是256)
#         transform = ResizeLongestSide(self.model.image_encoder.img_size)
        
#         # 转换为 numpy 图像
#         img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#         original_size = img_np.shape[:2]

#         print(f"📏 原始图像尺寸: {original_size}")
#         print(f"📏 模型目标尺寸: {self.model.image_encoder.img_size}")
        
#         # 官方预处理
#         input_image = transform.apply_image(img_np)
#         input_image_torch = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

#         print(f"📏 处理后图像tensor尺寸: {input_image_torch.shape}")

#         # 图像编码 
#         with torch.no_grad():
#             image_embedding = self.model.image_encoder(input_image_torch)

#         # 🔑 Prompt 编码：bbox 坐标变换
#         if input_boxes is not None:
#             box_np = np.array([input_boxes])
#             box_tf = transform.apply_boxes(box_np, original_size)
#             box = torch.tensor(box_tf, device=self.device).unsqueeze(0).float()
            
#             print(f"📦 原始 bbox: {input_boxes}")
#             print(f"📦 变换后 bbox: {box_tf[0]}")
#         else:
#             box = None

#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=None,
#             boxes=box,
#             masks=None,
#         )

#         # Mask 解码
#         low_res_masks, iou_predictions = self.model.mask_decoder(
#             image_embeddings=image_embedding,
#             image_pe=self.model.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )

#         # 后处理：还原到原图尺寸
#         pred_masks = torch.sigmoid(low_res_masks)
#         pred_masks = F.interpolate(
#             pred_masks,
#             size=original_size,
#             mode="bilinear",
#             align_corners=False
#         )

#         # 如果提供了 target_mask，计算 loss
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
#             if target_mask.shape != pred_masks.shape:
#                 target_mask = F.interpolate(target_mask, size=pred_masks.shape[-2:], mode="bilinear", align_corners=False)
#             loss = F.binary_cross_entropy(pred_masks, target_mask)
#             return pred_masks, loss

#         return pred_masks

# class SAMMed2DModel(SegmentationModelBase):
#     def __init__(self, device, cfg):
#         super().__init__(device, cfg)
#         from segment_anything import sam_model_registry
#         import os
    
#         model_type   = cfg.get("model_type", "vit_b")
#         ckpt_path    = cfg.get("local_path", "./pretrain_model/sam-med2d_b_clean.pth")
#         use_adapter  = cfg.get("encoder_adapter", True)
    
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
    
        
#         args = create_official_args(model_type, ckpt_path, use_adapter, device)
#         args.image_size = cfg.get("image_size", 256)
        
#         print(f"✅ SAM-Med2D 使用图像尺寸: {args.image_size}")
            
#         self.model = sam_model_registry[args.model_type](args).to(args.device)
#         # self.model = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
#         # self.model.eval()
#         print(f"✅ 成功加载 SAM-Med2D: {model_type} from {ckpt_path}")

#         print(f"⚠️ args.image_size = {args.image_size}")
#         print(f"⚠️ self.model.image_encoder.img_size = {self.model.image_encoder.img_size}")

#         # 强制所有模块进入训练模式
#         self.model.train()
#         self.model.image_encoder.train()
#         self.model.prompt_encoder.train()
#         self.model.mask_decoder.train()
        
#         # 解冻 image_encoder 的所有参数
#         for param in self.model.image_encoder.parameters():
#             param.requires_grad = True

#     def forward(self, img_tensor, input_boxes=None, target_mask=None):
#         # 删除所有numpy转换代码，直接使用PyTorch操作
#         from segment_anything.utils.transforms import ResizeLongestSide
#         print(f"📏 [SAMMed2D] 输入图像尺寸: {img_tensor.shape}")

#         # 🔑 关键修复：确保输入张量移到正确设备
#         img_tensor = img_tensor.to(self.device)
        
#         # 🔑 额外修复：确保target_mask也在正确设备上  
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
            
#         print(f"🔍 [SAMMed2D] 输入图像尺寸: {img_tensor.shape}")
#         print(f"🔍 [SAMMed2D] 输入设备: {img_tensor.device}")
#         print(f"🔍 [SAMMed2D] 模型设备: {self.device}")
        
#         # 直接使用原始tensor，避免numpy转换
#         original_size = img_tensor.shape[2:]  # [H, W]
        
#         # 创建纯PyTorch的预处理流程
#         transform = ResizeLongestSide(self.model.image_encoder.img_size)
        
#         # PyTorch实现的图像缩放 (替换apply_image)
#         scale = self.model.image_encoder.img_size / max(original_size)
#         new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)
#         resized_tensor = F.interpolate(
#             img_tensor, 
#             size=(new_h, new_w), 
#             mode="bilinear", 
#             align_corners=False
#         )
        
#         # PyTorch实现的padding (替换apply_image的padding)
#         h_pad = self.model.image_encoder.img_size - new_h
#         w_pad = self.model.image_encoder.img_size - new_w
#         input_image_torch = F.pad(
#             resized_tensor, 
#             (0, w_pad, 0, h_pad, 0, 0), 
#             value=0
#         )
        
#         print(f"📏 [SAMMed2D] 处理后图像尺寸: {input_image_torch.shape}")
        
#         # 图像编码 (保持梯度)
#         image_embedding = self.model.image_encoder(input_image_torch)
#         print(f"✅ image_embedding.requires_grad: {image_embedding.requires_grad}")
        
#         # 边界框处理 (纯PyTorch)
#         if input_boxes is not None:
#             # 边界框坐标缩放
#             box_tensor = torch.tensor([input_boxes], device=self.device).float()
#             box_tensor[:, 0::2] *= scale  # x坐标缩放
#             box_tensor[:, 1::2] *= scale  # y坐标缩放
#             print(f"📦 [SAMMed2D] 缩放后bbox: {box_tensor[0].tolist()}")

      	
#         # 🔑 关键修复6：完全复制test脚本的prompt编码
#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=None,
#             boxes=box_tensor,
#             masks=None,
#         )

#         # 🔑 关键修复7：完全复制test脚本的mask解码
#         low_res_masks, iou_predictions = self.model.mask_decoder(
#             image_embeddings=image_embedding,
#             image_pe=self.model.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )

#         # 🔑 关键修复8：完全复制test脚本的后处理
#         pred_masks = torch.sigmoid(low_res_masks)
        
#         # 注意：test脚本使用的是image.shape[:2]，这里要用original_size
#         pred_masks = F.interpolate(
#             pred_masks,
#             size=original_size,
#             mode="bilinear",
#             align_corners=False
#         )
        
#         print(f"📊 [SAMMed2D] 最终pred_masks形状: {pred_masks.shape}")
#         print(f"📊 [SAMMed2D] pred_masks范围: [{pred_masks.min().item():.3f}, {pred_masks.max().item():.3f}]")

#         # 如果提供了 target_mask，计算 loss
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
#             if target_mask.shape != pred_masks.shape:
#                 target_mask = F.interpolate(target_mask, size=pred_masks.shape[-2:], mode="bilinear", align_corners=False)
#             loss = F.binary_cross_entropy(pred_masks, target_mask)
#             return pred_masks, loss

#         return pred_masks

#     # def forward(self, img_tensor, input_boxes=None, target_mask=None):
#     #     """
#     #     修复版：与min_sammed2_test.py完全一致的处理逻辑
#     #     Args:
#     #         img_tensor: [B, C, H, W] in [0, 1]
#     #         input_boxes: list of [x1, y1, x2, y2] (原始坐标)
#     #         target_mask: [B, 1, H, W] 用于计算 loss
#     #     Returns:
#     #         pred_masks: [B, 1, H, W]
#     #         loss: 如果提供了 target_mask
#     #     """
        
#     #     from segment_anything.utils.transforms import ResizeLongestSide
        
#     #     # 🔑 关键修复1：确保使用正确的image_size
#     #     transform = ResizeLongestSide(self.model.image_encoder.img_size)
#     #     print(f"⚠️ ResizeLongestSide使用的尺寸: {self.model.image_encoder.img_size}")
#     #     # exit(0)
        
#     #     # 🔑 关键修复2：精确复制test脚本的数据转换
#     #     # 避免多次转换导致的精度损失
#     #     if img_tensor.dim() == 4:  # [B, C, H, W]
#     #         img_np = (img_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
#     #     else:  # [C, H, W]
#     #         img_np = (img_tensor.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        
#     #     original_size = img_np.shape[:2]
        
#     #     print(f"📏 [SAMMed2D] 原始图像尺寸: {original_size}")
#     #     print(f"📏 [SAMMed2D] 模型目标尺寸: {self.model.image_encoder.img_size}")
#     #     print(f"📏 [SAMMed2D] 图像数据范围: [{img_np.min()}, {img_np.max()}]")
        
#     #     # 🔑 关键修复3：完全复制test脚本的预处理流程
#     #     input_image = transform.apply_image(img_np)
#     #     input_image_torch = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

#     #     print(f"📏 [SAMMed2D] 处理后图像tensor尺寸: {input_image_torch.shape}")
#     #     print(f"📏 [SAMMed2D] 处理后tensor范围: [{input_image_torch.min().item():.3f}, {input_image_torch.max().item():.3f}]")
				
#     #     # with torch.no_grad():
#     #     image_embedding = self.model.image_encoder(input_image_torch)
#     #     print("image_embedding.requires_grad:", image_embedding.requires_grad)
				
#     #     if input_boxes is not None:
#     #         # box_np = np.array([input_boxes])
#     #         # box_tf = transform.apply_boxes(box_np, original_size)
#     #         # box_torch = torch.tensor(box_tf, device=self.device).unsqueeze(0).float()
            
#     #         box_torch = torch.tensor([input_boxes], device=self.device).unsqueeze(0).float()
#     #         print(f"📦 [SAMMed2D] 原始 bbox: {input_boxes}")
#     #         # print(f"📦 [SAMMed2D] 变换后 bbox: {box_tf[0]}")
#     #     else:
#     #         box_torch = None

#     #     # 🔑 关键修复6：完全复制test脚本的prompt编码
#     #     sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#     #         points=None,
#     #         boxes=box_torch,
#     #         masks=None,
#     #     )

#     #     # 🔑 关键修复7：完全复制test脚本的mask解码
#     #     low_res_masks, iou_predictions = self.model.mask_decoder(
#     #         image_embeddings=image_embedding,
#     #         image_pe=self.model.prompt_encoder.get_dense_pe(),
#     #         sparse_prompt_embeddings=sparse_embeddings,
#     #         dense_prompt_embeddings=dense_embeddings,
#     #         multimask_output=False,
#     #     )

#     #     # 🔑 关键修复8：完全复制test脚本的后处理
#     #     pred_masks = torch.sigmoid(low_res_masks)
        
#     #     # 注意：test脚本使用的是image.shape[:2]，这里要用original_size
#     #     pred_masks = F.interpolate(
#     #         pred_masks,
#     #         size=original_size,
#     #         mode="bilinear",
#     #         align_corners=False
#     #     )
        
#     #     print(f"📊 [SAMMed2D] 最终pred_masks形状: {pred_masks.shape}")
#     #     print(f"📊 [SAMMed2D] pred_masks范围: [{pred_masks.min().item():.3f}, {pred_masks.max().item():.3f}]")

#     #     # 如果提供了 target_mask，计算 loss
#     #     if target_mask is not None:
#     #         target_mask = target_mask.to(self.device)
#     #         if target_mask.shape != pred_masks.shape:
#     #             target_mask = F.interpolate(target_mask, size=pred_masks.shape[-2:], mode="bilinear", align_corners=False)
#     #         loss = F.binary_cross_entropy(pred_masks, target_mask)
#     #         return pred_masks, loss

#     #     return pred_masks

#     def predict(self, img_tensor, input_boxes=None):
#         """推理模式，只返回预测结果，保持与父类接口一致"""
#         return self.forward(img_tensor, input_boxes)

# # ---------------------------- SAM-Med2D ----------------------------
# class SAMMed2DModel(SegmentationModelBase):
#     def __init__(self, device, cfg):
#         super().__init__(device, cfg)
#         from segment_anything import sam_model_registry
#         import os
        
#         # 加载配置
#         model_type = cfg.get("model_type", "vit_b")
#         ckpt_path = cfg.get("local_path", "./pretrain_model/sam-med2d_b.pth")
#         use_adapter = cfg.get("encoder_adapter", True)
        
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
        
#         # 创建官方格式的args对象
#         args = create_official_args(model_type, ckpt_path, use_adapter, device)
#         args.image_size = cfg.get("image_size", 256)
        
#         print(f"✅ SAM-Med2D 配置: 模型={model_type}, 尺寸={args.image_size}, 适配器={use_adapter}")
        
#         # 加载模型（关键修复：保持梯度）
#         self.model = sam_model_registry[args.model_type](args).to(args.device)
#         self.model.train()  # ✅ 必须设置为训练模式以支持对抗攻击
#         print(f"✅ 成功加载 SAM-Med2D: {ckpt_path}")

#     def forward(self, img_tensor, input_boxes=None, target_mask=None):
#         """
#         完全复用之前成功的实现逻辑
#         """
#         from segment_anything.utils.transforms import ResizeLongestSide
        
#         # 确保设备一致
#         img_tensor = img_tensor.to(self.device)
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
        
#         # 获取原始尺寸
#         original_size = img_tensor.shape[2:]  # (H, W)
        
#         # 官方预处理
#         transform = ResizeLongestSide(self.model.image_encoder.img_size)
#         scale = self.model.image_encoder.img_size / max(original_size)
#         new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)
        
#         # 缩放
#         resized_tensor = F.interpolate(img_tensor, size=(new_h, new_w), 
#                                      mode="bilinear", align_corners=False)
        
#         # Padding
#         h_pad = self.model.image_encoder.img_size - new_h
#         w_pad = self.model.image_encoder.img_size - new_w
#         input_image_torch = F.pad(resized_tensor, (0, w_pad, 0, h_pad, 0, 0), value=0)
        
#         # 图像编码（保持梯度）
#         image_embedding = self.model.image_encoder(input_image_torch)
        
#         # 边界框处理
#         box_tensor = None
#         if input_boxes is not None:
#             box_tensor = torch.tensor([input_boxes], device=self.device).float()
#             box_tensor[:, 0::2] *= scale  # x坐标缩放
#             box_tensor[:, 1::2] *= scale  # y坐标缩放
        
#         # Prompt编码
#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=None, boxes=box_tensor, masks=None
#         )
        
#         # Mask解码
#         low_res_masks, iou_predictions = self.model.mask_decoder(
#             image_embeddings=image_embedding,
#             image_pe=self.model.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )
        
#         # 后处理
#         pred_masks = torch.sigmoid(low_res_masks)
#         pred_masks = F.interpolate(pred_masks, size=original_size, 
#                                  mode="bilinear", align_corners=False)
        
#         # 计算损失
#         if target_mask is not None:
#             if target_mask.shape != pred_masks.shape:
#                 target_mask = F.interpolate(target_mask, size=pred_masks.shape[-2:], 
#                                           mode="bilinear", align_corners=False)
#             loss = F.binary_cross_entropy(pred_masks, target_mask)
#             return pred_masks, loss
        
#         return pred_masks

#     def predict(self, img_tensor, input_boxes=None):
#         """推理模式"""
#         return self.forward(img_tensor, input_boxes)

# ---------------------------- SAM-Med2D ----------------------------
# class SAMMed2DModel(SegmentationModelBase):
#     def __init__(self, device, cfg):
#         super().__init__(device, cfg)
#         from segment_anything import sam_model_registry
#         import os
        
#         # 加载配置
#         model_type = cfg.get("model_type", "vit_b")
#         ckpt_path = cfg.get("local_path", "./pretrain_model/sam-med2d_b.pth")
#         use_adapter = cfg.get("encoder_adapter", True)
        
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
        
#         # 创建官方格式的args对象
#         args = create_official_args(model_type, ckpt_path, use_adapter, device)
#         args.image_size = cfg.get("image_size", 256)
        
#         print(f"✅ SAM-Med2D 配置: 模型={model_type}, 尺寸={args.image_size}, 适配器={use_adapter}")
        
#         # 加载模型并保持训练模式（用于对抗攻击）
#         self.model = sam_model_registry[args.model_type](args).to(args.device)
#         self.model.train()
#         print(f"✅ 成功加载 SAM-Med2D: {ckpt_path}")

#     def forward(self, img_tensor, input_boxes=None, target_mask=None):
#         """
#         修复版：直接使用预处理后的输入，避免双重预处理
#         """
#         from segment_anything.utils.transforms import ResizeLongestSide
        
#         # 确保设备一致
#         img_tensor = img_tensor.to(self.device)
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
        
#         # 获取尺寸（应该是256x256）
#         _, _, H, W = img_tensor.shape
#         original_size = (H, W)
        
#         # ✅ 关键修复：跳过重复预处理，直接使用输入（值范围已是[0,1]）
#         input_image_torch = img_tensor
        
#         # 图像编码（保持梯度用于对抗攻击）   ）
#         image_embedding = self.model.image_encoder(input_image_torch)
        
#         # ✅ 修复：正确处理bbox格式 [1, 1, 4]
#         box_tensor = None
#         if input_boxes is not None:
#             if isinstance(input_boxes, (list, tuple, np.ndarray)):
#                 box_tensor = torch.tensor([[input_boxes]], device=self.device).float()  # [1, 1, 4]
#             else:
#                 box_tensor = input_boxes.float().to(self.device)
#                 if box_tensor.dim() == 2:
#                     box_tensor = box_tensor.unsqueeze(0)  # [1, 1, 4]
        
#         # Prompt编码
#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=None, boxes=box_tensor, masks=None
#         )
        
#         # Mask解码
#         low_res_masks, iou_predictions = self.model.mask_decoder(
#             image_embeddings=image_embedding,
#             image_pe=self.model.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )
        
#         # 后处理
#         pred_masks = torch.sigmoid(low_res_masks)
#         pred_masks = F.interpolate(pred_masks, size=original_size, 
#                                  mode="bilinear", align_corners=False)
        
#         # 计算损失
#         if target_mask is not None:
#             if target_mask.shape != pred_masks.shape:
#                 target_mask = F.interpolate(target_mask, size=pred_masks.shape[-2:], 
#                                           mode="bilinear", align_corners=False)
#             loss = F.binary_cross_entropy(pred_masks, target_mask)
#             return pred_masks, loss
        
#         return pred_masks

#     def predict(self, img_tensor, input_boxes=None):
#         """推理模式"""
#         return self.forward(img_tensor, input_boxes)

# class SAMMed2DModel(SegmentationModelBase):
#     def __init__(self, device, cfg):
#         super().__init__(device, cfg)
#         from segment_anything import sam_model_registry
        
#         # 加载配置（保持不变）
#         model_type = cfg.get("model_type", "vit_b")
#         ckpt_path = cfg.get("local_path", "./pretrain_model/sam-med2d_b.pth")
#         use_adapter = cfg.get("encoder_adapter", True)
        
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
        
#         # ✅ 加载模型（必须train模式支持对抗攻击梯度）
#         args = create_official_args(model_type, ckpt_path, use_adapter, device)
#         args.image_size = cfg.get("image_size", 256)
#         self.model = sam_model_registry[args.model_type](args).to(args.device)
#         self.model.train()  # ✅ 关键：训练模式
#         self.image_size = args.image_size
        
#         print(f"✅ 成功加载 SAM-Med2D: {model_type}, 图像尺寸: {self.image_size}")
        
#         # ✅ 初始化官方预处理（用于内部坐标变换）
#         from segment_anything.utils.transforms import ResizeLongestSide
#         self.transform = ResizeLongestSide(self.image_size)

#     def forward(self, img_tensor, input_boxes=None, target_mask=None):
#         """
#         ✅ 完全复现SammedPredictor逻辑，但纯PyTorch实现保持梯度
#         """
#         # 设备同步
#         img_tensor = img_tensor.to(self.device)
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
        
#         # ✅ 获取原始尺寸（应为256x256）
#         batch_size, _, H, W = img_tensor.shape
#         original_size = (H, W)
        
#         # ✅ 将[0,1]tensor转换为uint8 numpy（与独立实验一致）
#         # 注意：这步会丢失梯度，但SammedPredictor内部就是这么做的
#         # 关键在于后续所有操作都是PyTorch
#         img_np = (img_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        
#         # ✅ 复现set_image()的预处理
#         input_image = self.transform.apply_image(img_np)
#         input_image_torch = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
#         # ✅ 图像编码（保持梯度）
#         image_embedding = self.model.image_encoder(input_image_torch)
        
#         # ✅ 复现predict()的bbox处理（格式：[1, 1, 4]）
#         box_tensor = None
#         if input_boxes is not None:
#             # 坐标缩放（因为input_image_torch尺寸可能>256）
#             scale = self.image_size / max(original_size)
            
#             # 统一转换为numpy
#             if isinstance(input_boxes, (list, tuple)):
#                 box_np = np.array(input_boxes)
#             elif isinstance(input_boxes, np.ndarray):
#                 box_np = input_boxes
#             else:  # tensor
#                 box_np = input_boxes.detach().cpu().numpy()
            
#             # 应用官方变换（缩放+padding）
#             box_tf = self.transform.apply_boxes(box_np.reshape(1, 4), original_size)
#             box_tensor = torch.tensor(box_tf, device=self.device).unsqueeze(0).float()
        
#         # ✅ Prompt编码
#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=None, boxes=box_tensor, masks=None
#         )
        
#         # ✅ Mask解码
#         low_res_masks, iou_predictions = self.model.mask_decoder(
#             image_embeddings=image_embedding,
#             image_pe=self.model.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )
        
#         # ✅ 后处理（sigmoid + 插值回原尺寸）
#         pred_masks = torch.sigmoid(low_res_masks)
#         pred_masks = F.interpolate(pred_masks, size=original_size, 
#                                  mode="bilinear", align_corners=False)
        
#         # ✅ 计算对抗攻击所需loss
#         if target_mask is not None:
#             if target_mask.shape != pred_masks.shape:
#                 target_mask = F.interpolate(target_mask, size=pred_masks.shape[-2:], 
#                                           mode="bilinear", align_corners=False)
#             loss = F.binary_cross_entropy(pred_masks, target_mask)
#             return pred_masks, loss
        
#         return pred_masks
# model_zoo.py - 只修改 SAMMed2DModel 类
# model_zoo.py - 修复 SAM-Med2D 部分
# class SAMMed2DModel(SegmentationModelBase):
#     def __init__(self, device, cfg):
#         super().__init__(device, cfg)
#         from segment_anything import sam_model_registry
#         from segment_anything.predictor_sammed import SammedPredictor
        
#         # 加载配置
#         model_type = cfg.get("model_type", "vit_b")
#         ckpt_path = cfg.get("local_path", "./pretrain_model/sam-med2d_b.pth")
#         image_size = cfg.get("image_size", 256)
#         use_adapter = cfg.get("encoder_adapter", True)
        
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
        
#         # ✅ 1. 创建与独立实现完全一致的 args
#         args = argparse.Namespace(
#             image_size=image_size,
#             encoder_adapter=use_adapter,
#             sam_checkpoint=ckpt_path,
#             model_type=model_type,
#             device=device
#         )
        
#         # ✅ 2. 使用 sam_model_registry 加载模型（与独立实现一致）
#         self.model = sam_model_registry[model_type](args).to(device)
#         self.model.eval()  # ✅ 关键：设置为评估模式
        
#         # ✅ 3. 创建官方 predictor（与独立实现完全一致）
#         self.predictor = SammedPredictor(self.model)
#         self.image_size = image_size
        
#         print(f"✅ 成功加载 SAM-Med2D: {model_type}, 图像尺寸: {image_size}")
#         print(f"✅ 使用官方 SammedPredictor，与独立测试实现完全一致")

#     def forward(self, img_tensor, input_boxes=None, target_mask=None):
#         """
#         Args:
#             img_tensor: [B, C, H, W] in [0, 1]
#             input_boxes: list of [x1, y1, x2, y2] (原始坐标)
#             target_mask: [B, 1, H, W] 用于计算 loss
#         Returns:
#             pred_masks: [B, 1, H, W]
#             loss: 如果提供了 target_mask
#         """
#         # ✅ 设备同步（对抗攻击时需要）
#         img_tensor = img_tensor.to(self.device)
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
        
#         # 获取批次和尺寸
#         batch_size, _, H, W = img_tensor.shape
#         original_size = (H, W)
        
#         # ✅ 4. 复现 predictor.set_image() 的完整逻辑
#         # 将 [0,1] tensor 转换为 uint8 numpy（与独立实现一致）
#         img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
#         # ✅ 5. 调用 predictor.set_image()（自动完成所有预处理）
#         self.predictor.set_image(img_np)
        
#         # ✅ 6. 准备 bbox 格式（predictor.predict 需要的格式）
#         box_input = np.array([input_boxes]) if input_boxes is not None else None
        
#         # ✅ 7. 调用 predictor.predict()（与独立实现完全一致）
#         masks, iou_predictions, low_res_logits = self.predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=box_input,
#             multimask_output=False
#         )
        
#         # masks 是 numpy array，需要转换为 tensor
#         pred_masks = torch.from_numpy(masks).float().to(self.device)
#         pred_masks = pred_masks.unsqueeze(0)  # [1, 1, H, W]
        
#         # ✅ 8. 计算对抗攻击所需的 loss
#         if target_mask is not None:
#             # 确保尺寸匹配
#             if pred_masks.shape != target_mask.shape:
#                 target_mask = F.interpolate(target_mask, size=pred_masks.shape[-2:], 
#                                           mode="bilinear", align_corners=False)
#             loss = F.binary_cross_entropy(pred_masks, target_mask)
#             return pred_masks, loss
        
#         return pred_masks
    
#     def predict(self, img_tensor, input_boxes=None):
#         """推理模式"""
#         return self.forward(img_tensor, input_boxes)
# model_zoo.py - 只替换 SAMMed2DModel 类

# class SAMMed2DModel(SegmentationModelBase):
#     def __init__(self, device, cfg):
#         super().__init__(device, cfg)
#         from segment_anything import sam_model_registry
#         from segment_anything.utils.transforms import ResizeLongestSide
        
#         # 加载配置（保持不变）
#         model_type = cfg.get("model_type", "vit_b")
#         ckpt_path = cfg.get("local_path", "./pretrain_model/sam-med2d_b.pth")
#         image_size = cfg.get("image_size", 256)
#         use_adapter = cfg.get("encoder_adapter", True)
        
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
        
#         # ✅ 创建args（与独立实现一致）
#         args = argparse.Namespace(
#             image_size=image_size,
#             encoder_adapter=use_adapter,
#             sam_checkpoint=ckpt_path,
#             model_type=model_type,
#             device=device
#         )
        
#         # ✅ 加载模型
#         self.model = sam_model_registry[model_type](args).to(device)
#         self.model.eval()  # 默认评估模式
        
#         # ✅ 关键：初始化官方变换器（用于bbox坐标缩放）
#         self.transform = ResizeLongestSide(image_size)
#         self.image_size = image_size
        
#         print(f"✅ 成功加载 SAM-Med2D: {model_type}, 图像尺寸: {image_size}")
#         print(f"✅ 使用纯PyTorch实现，保持梯度流用于对抗攻击")

#     def forward(self, img_tensor, input_boxes=None, target_mask=None):
#         """
#         ✅ 纯PyTorch实现，逐行对照 SammedPredictor 的 set_image() 和 predict()
#         """
#         # ✅ 设备同步
#         img_tensor = img_tensor.to(self.device)
#         if target_mask is not None:
#             target_mask = target_mask.to(self.device)
        
#         # 获取尺寸
#         batch_size, _, H, W = img_tensor.shape
#         original_size = (H, W)
        
#         # ============================================================
#         # ✅ 阶段1：图像预处理（对应 SammedPredictor.set_image()）
#         # ============================================================
#         # 缩放逻辑：ResizeLongestSide.apply_image()
#         scale = self.image_size / max(original_size)
#         new_h, new_w = int(H * scale), int(W * scale)
#         resized_tensor = F.interpolate(
#             img_tensor, 
#             size=(new_h, new_w), 
#             mode="bilinear", 
#             align_corners=False  # 与官方一致
#         )
        
#         # Padding逻辑：ResizeLongestSide.apply_image()
#         h_pad = self.image_size - new_h
#         w_pad = self.image_size - new_w
#         input_image_torch = F.pad(resized_tensor, (0, w_pad, 0, h_pad, 0, 0), value=0)
        
#         # ✅ 图像编码（对应 SammedPredictor.set_image() 的 model.image_encoder）
#         image_embedding = self.model.image_encoder(input_image_torch)
        
#         # ============================================================
#         # ✅ 阶段2：Prompt处理（对应 SammedPredictor.predict()）
#         # ============================================================
#         # ✅ 关键：bbox坐标变换必须使用官方 transform.apply_boxes()
#         # 这是IOU精度的核心保证！
#         box_tensor = None
#         if input_boxes is not None:
#             # 转换为numpy（因为apply_boxes期望numpy输入）
#             if isinstance(input_boxes, (list, tuple, np.ndarray)):
#                 box_np = np.array(input_boxes).reshape(1, 4)
#             else:  # tensor
#                 box_np = input_boxes.detach().cpu().numpy().reshape(1, 4)
            
#             # ✅ 使用官方变换（精确复现）
#             box_tf = self.transform.apply_boxes(box_np, original_size)
#             box_tensor = torch.tensor(box_tf, device=self.device).unsqueeze(0).float()
        
#         # ✅ Prompt编码（与官方完全一致）
#         sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
#             points=None, 
#             boxes=box_tensor, 
#             masks=None
#         )
        
#         # ============================================================
#         # ✅ 阶段3：Mask解码与后处理（对应 SammedPredictor.predict()）
#         # ============================================================
#         low_res_masks, iou_predictions = self.model.mask_decoder(
#             image_embeddings=image_embedding,
#             image_pe=self.model.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )
        
#         # ✅ 后处理（sigmoid + 插值到原图尺寸）
#         pred_masks = torch.sigmoid(low_res_masks)
#         pred_masks = F.interpolate(
#             pred_masks, 
#             size=original_size, 
#             mode="bilinear", 
#             align_corners=False
#         )
        
#         # ============================================================
#         # ✅ 阶段4：对抗攻击损失计算
#         # ============================================================
#         if target_mask is not None:
#             if target_mask.shape != pred_masks.shape:
#                 target_mask = F.interpolate(
#                     target_mask, 
#                     size=pred_masks.shape[-2:], 
#                     mode="bilinear", 
#                     align_corners=False
#                 )
#             loss = F.binary_cross_entropy(pred_masks, target_mask)
#             return pred_masks, loss
        
#         return pred_masks
    
#     def predict(self, img_tensor, input_boxes=None):
#         """推理模式（保持接口一致）"""
#         return self.forward(img_tensor, input_boxes)
# model_zoo.py - 直接替换整个 SAMMed2DModel 类


class SAMMed2DModel(SegmentationModelBase):
    def __init__(self, device, cfg):
        super().__init__(device, cfg)
        from segment_anything import sam_model_registry
        from segment_anything.utils.transforms import ResizeLongestSide
        
        # 加载配置
        model_type = cfg.get("model_type", "vit_b")
        ckpt_path = cfg.get("local_path", "./pretrain_model/sam-med2d_b.pth")
        image_size = cfg.get("image_size", 256)
        use_adapter = cfg.get("encoder_adapter", True)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ SAM-Med2D 权重未找到: {ckpt_path}")
        
        # 创建args（与独立测试脚本一致）
        args = argparse.Namespace(
            image_size=image_size,
            encoder_adapter=use_adapter,
            sam_checkpoint=ckpt_path,
            model_type=model_type,
            device=device
        )
        
        # 加载模型
        self.model = sam_model_registry[model_type](args).to(device)
        self.model.eval()  # 评估模式，但在对抗攻击时会临时切换到训练模式
        
        # 初始化官方变换器（用于bbox坐标缩放）
        self.transform = ResizeLongestSide(image_size)
        self.image_size = image_size
        
        # ✅ 修复1：确保pixel_mean/std在正确的设备上初始化
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(-1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(-1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
        
        print(f"✅ 成功加载 SAM-Med2D: {model_type}, 图像尺寸: {image_size}")
        print(f"✅ 已注册正确的归一化参数，与独立测试脚本保持一致")

    def forward(self, img_tensor, input_boxes=None, target_mask=None):
        """
        ✅ 纯PyTorch实现，正确处理归一化，保持梯度流用于对抗攻击
        """
        # ✅ 修复2：确保所有张量在同一设备上
        img_tensor = img_tensor.to(self.device)
        if target_mask is not None:
            target_mask = target_mask.to(self.device)
        
        # 获取尺寸
        batch_size, _, H, W = img_tensor.shape
        original_size = (H, W)
        
        # ============================================================
        # ✅ 阶段1：图像预处理
        # ============================================================
        # 缩放逻辑
        scale = self.image_size / max(original_size)
        new_h, new_w = int(H * scale), int(W * scale)
        resized_tensor = F.interpolate(
            img_tensor, 
            size=(new_h, new_w), 
            mode="bilinear", 
            align_corners=False
        )
        
        # Padding逻辑
        h_pad = self.image_size - new_h
        w_pad = self.image_size - new_w
        input_image_torch = F.pad(resized_tensor, (0, w_pad, 0, h_pad, 0, 0), value=0)
        
        # ✅ 关键：应用正确的归一化
        # 将[0,1]范围转换为[0,255]，再进行归一化
        if input_image_torch.max() <= 1.0:
            input_image_torch = input_image_torch * 255.0
        
        # ✅ 修复3：确保pixel_mean/std在正确的设备上
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std
        
        # ✅ 图像编码
        image_embedding = self.model.image_encoder(input_image_torch)
        
        # ============================================================
        # ✅ 阶段2：Prompt处理
        # ============================================================
        # ✅ 关键：bbox坐标变换必须使用官方 transform.apply_boxes()
        box_tensor = None
        if input_boxes is not None:
            # 转换为numpy（因为apply_boxes期望numpy输入）
            if isinstance(input_boxes, (list, tuple, np.ndarray)):
                box_np = np.array(input_boxes).reshape(1, 4)
            else:  # tensor
                box_np = input_boxes.detach().cpu().numpy().reshape(1, 4)
            # ✅ 使用官方变换（精确复现）
            box_tf = self.transform.apply_boxes(box_np, original_size)
            box_tensor = torch.tensor(box_tf, device=self.device).unsqueeze(0).float()
        
        # ✅ Prompt编码
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None, 
            boxes=box_tensor, 
            masks=None
        )
        
        # ============================================================
        # ✅ 阶段3：Mask解码与后处理
        # ============================================================
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        # ✅ 后处理：先插值logits到原图尺寸，再应用sigmoid
        # 保留logits用于loss计算（BCE_with_logits更稳定，支持autocast）
        low_res_masks_resized = F.interpolate(
            low_res_masks, 
            size=original_size, 
            mode="bilinear", 
            align_corners=False
        )
        
        # 应用sigmoid得到最终预测
        pred_masks = torch.sigmoid(low_res_masks_resized)
        
        # ============================================================
        # ✅ 阶段4：对抗攻击损失计算（使用logits，支持mixed precision）
        # ============================================================
        if target_mask is not None:
            if target_mask.shape != pred_masks.shape:
                target_mask = F.interpolate(
                    target_mask, 
                    size=pred_masks.shape[-2:], 
                    mode="bilinear", 
                    align_corners=False
                )
            # ✅ 使用binary_cross_entropy_with_logits（autocast安全）
            loss = F.binary_cross_entropy_with_logits(low_res_masks_resized, target_mask)
            return pred_masks, loss
        
        return pred_masks

    def predict(self, img_tensor, input_boxes=None):
        """推理模式（保持接口一致）"""
        with torch.no_grad():
            return self.forward(img_tensor, input_boxes)

# ---------------------------- 模型工厂 ----------------------------
MODEL_REGISTRY = {
    "medsam": MedSAMModel,
    "sam": SAMModel,
  	"sammed2d": SAMMed2DModel,
    "sam_point": SAMPointModel,
    "unet": UNetModel,
    "deeplab": DeepLabV3Model,
}

def create_model(model_name: str, device: str, model_cfg: dict) -> SegmentationModelBase:
    """模型工厂函数"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {model_name}，支持的模型: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class(device, model_cfg)

def list_available_models():
    """列出所有可用模型"""
    return list(MODEL_REGISTRY.keys())
