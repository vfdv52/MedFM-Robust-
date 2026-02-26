# OK：isic，mri，breast，内窥镜，视网膜暂时认为OK, 病理OK
# for segmentation data
# python segmentation_generate_perb_all_V9_adpative_efficient.py --dataset_name andrewmvd-cancer-inst --adaptive
# 给不同扰动水平设置特定的精度level，最简单的方式就是根据目前的极限施压后的精度范围设定！ Haha
# python segmentation_generate_perb_all_V9_adpative_efficient.py --all_datasets --adaptive

# ============================================================
# 🎚️  新功能：Level强度乘数控制 (--level_multipliers)
# ============================================================
# 
# 使用方法：
# ---------
# 通过 --level_multipliers 参数可以轻松调整特定level的扰动强度，无需手动修改代码！
# 
# 基本用法：
# python segmentation_generate_perb_all_V9_adpative_efficient.py \
#     --dataset_name isic_2016 \
#     --level_multipliers '{"1": 2.0, "3": 3.0, "5": 4.0}'
# 
# 说明：
# - 上面的命令会让 level1 强度×2, level3 强度×3, level5 强度×4
# - 未指定的level保持默认强度（乘数为1.0）
# - 乘数范围：0.1 到 10.0
# 
# 常用场景示例：
# --------------
# 
# 1. 增强所有level（一般情况）：
# --level_multipliers '{"1": 1.5, "2": 2.0, "3": 2.5, "4": 3.0, "5": 4.0}'
# 
# 2. 只增强你使用的level（推荐）：
# --level_multipliers '{"1": 2.0, "3": 3.0, "5": 4.0}'
# 
# 3. 极限测试：
# --level_multipliers '{"1": 3.0, "3": 5.0, "5": 8.0}'
# 
# 4. 轻微增强：
# --level_multipliers '{"1": 1.2, "3": 1.5, "5": 2.0}'
# 
# 完整示例：
# ----------
# # 为所有数据集生成增强扰动
# python segmentation_generate_perb_all_V9_adpative_efficient.py \
#     --all_datasets \
#     --level_multipliers '{"1": 2.0, "3": 3.0, "5": 4.0}'
# 
# # 为单个数据集生成，并启用adaptive模式
# python segmentation_generate_perb_all_V9_adpative_efficient.py \
#     --dataset_name isic_2016 \
#     --adaptive \
#     --level_multipliers '{"1": 2.0, "3": 3.0, "5": 4.0}'
# 
# 注意事项：
# ---------
# - 不同参数类型的增强方式：
#   * 噪声/模糊类：直接乘以乘数（越大越强）
#   * 压缩质量：反向计算（质量下降越多扰动越强）
#   * Pixelate/Scale：反向计算（值越小扰动越强）
# - 建议先用保守的乘数（1.5-2.0）测试，确认效果后再提高
# - 可以单独查看生成的CSV文件中的SSIM值来评估扰动强度
# ============================================================

import numpy as np
import SimpleITK as sitk
import cv2
import os
import glob
import json
import csv
from pathlib import Path
import warnings
import random
import argparse
from typing import Dict, Any, List, Tuple, Optional
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.metrics import structural_similarity as ssim

class Medical2DPerturbationGenerator:
    """
    专为2D医学图像设计的扰动生成器
    支持参数缓存：第一张图自适应搜索，后续图片直接复用参数
    """
    
    def __init__(self, seed: int = 42, ssim_targets: Dict[int, Tuple[float, float]] = None,
                 level_multipliers: Dict[int, float] = None):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        self._init_modality_mapping()
        self._init_severity_parameters()
        
        # ========== Level强度乘数配置 ==========
        # 为每个level设置全局强度乘数，默认都是1.0（不变）
        # 例如: {1: 2.0, 3: 3.0, 5: 4.0} 表示level1强度×2, level3强度×3, level5强度×4
        self.level_multipliers = level_multipliers if level_multipliers is not None else {
            1: 1.0,  # level 1 强度乘数（默认不变）
            2: 1.0,  # level 2 强度乘数（默认不变）
            3: 1.0,  # level 3 强度乘数（默认不变）
            4: 1.0,  # level 4 强度乘数（默认不变）
            5: 1.0   # level 5 强度乘数（默认不变）
        }
        
        # 应用level乘数到所有参数
        self._apply_level_multipliers()
        
        self.ssim_targets = ssim_targets
        
        self.default_ssim_targets = {
            1: (0.90, 0.98),
            2: (0.80, 0.89),
            3: (0.70, 0.79),
            4: (0.60, 0.69),
            5: (0.50, 0.59)
        }
        
        # ========== 参数缓存机制 ==========
        # {perturbation_type: {severity: param_value}}
        self.cached_params: Dict[str, Dict[int, float]] = {}
    
    def clear_param_cache(self):
        """清空参数缓存（处理新数据集时调用）"""
        self.cached_params = {}
    
    def get_cached_param(self, perturbation_type: str, severity: int) -> Optional[float]:
        """获取缓存的参数值"""
        if perturbation_type in self.cached_params:
            return self.cached_params[perturbation_type].get(severity)
        return None
    
    def set_cached_param(self, perturbation_type: str, severity: int, param_value: float):
        """设置缓存的参数值"""
        if perturbation_type not in self.cached_params:
            self.cached_params[perturbation_type] = {}
        self.cached_params[perturbation_type][severity] = param_value
    
    def _init_modality_mapping(self):
        self.dataset_to_modality = {
            'isic': 'DERM', 'ISIC': 'DERM', 'skin': 'DERM', 'lesion': 'DERM', 'melanoma': 'DERM',
            'hyper-kvasir': 'ENDOSCOPY', 'kvasir': 'ENDOSCOPY', 'gastro': 'ENDOSCOPY', 
            'gi': 'ENDOSCOPY', 'endoscopy': 'ENDOSCOPY', 'polyp': 'ENDOSCOPY',
            'brain': 'MRI_BRAIN', 'tumor': 'MRI_BRAIN', 'mri_brain': 'MRI_BRAIN', 'brats': 'MRI_BRAIN',
            'breast': 'US_BREAST', 'ultrasound': 'US_BREAST', 'us_breast': 'US_BREAST', 
            'aryashah': 'US_BREAST', 'busi': 'US_BREAST',
            'cancer': 'PATHOLOGY', 'pathology': 'PATHOLOGY', 'histopathology': 'PATHOLOGY', 
            'nuclei': 'PATHOLOGY',
            'glaucoma': 'OCT', 'disc': 'OCT', 'cup': 'OCT', 'retina': 'OCT', 
            'fundus': 'OCT', 'ophthalmology': 'OCT',
            'ct': 'CT_ABDOMEN', 'abdomen': 'CT_ABDOMEN', 'flare': 'CT_ABDOMEN', 'miccai': 'CT_ABDOMEN'
        }
    
    def _init_severity_parameters(self):
        self.severity_params = {
            'gaussian_noise': {1: 0.01, 2: 0.02, 3: 0.03, 4: 0.05, 5: 0.08},
            'salt_pepper_noise': {1: 0.001, 2: 0.003, 3: 0.005, 4: 0.01, 5: 0.02},
            'speckle_noise': {1: 0.05, 2: 0.1, 3: 0.15, 4: 0.2, 5: 0.25},
            'gaussian_blur': {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0},
            'motion_blur': {1: 3, 2: 5, 3: 7, 4: 9, 5: 11},
            'brightness_contrast': {1: (0.05, 0.05), 2: (0.1, 0.1), 3: (0.15, 0.15), 
                                  4: (0.2, 0.2), 5: (0.3, 0.3)},
            'compression_artifacts': {1: 90, 2: 70, 3: 50, 4: 30, 5: 10},
            'pixelate': {1: 0.8, 2: 0.6, 3: 0.5, 4: 0.4, 5: 0.3},
            'rotation': {1: 5, 2: 10, 3: 15, 4: 20, 5: 25},
            'scale': {1: 0.95, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75},
            'translation': {1: 5, 2: 10, 3: 15, 4: 20, 5: 25},
            
            # ===== MRI特定参数 =====
            'mri_motion_translation': {1: 2.0, 2: 5.0, 3: 10.0, 4: 18.0, 5: 30.0},
            'mri_motion_corrupted_pct': {1: 0.15, 2: 0.30, 3: 0.45, 4: 0.60, 5: 0.80},
            'mri_bias_intensity': {1: 0.30, 2: 0.50, 3: 0.70, 4: 0.85, 5: 0.95},
            'mri_ghost_intensity': {1: 0.10, 2: 0.20, 3: 0.35, 4: 0.50, 5: 0.70},
            'mri_ghost_count': {1: 2, 2: 3, 3: 4, 4: 5, 5: 6},
            
            # ===== 超声特定参数 =====
            'us_shadow_attenuation': {1: 0.30, 2: 0.45, 3: 0.55, 4: 0.70, 5: 0.85},
            'us_reverb_intensity': {1: 0.12, 2: 0.18, 3: 0.25, 4: 0.35, 5: 0.45},
            
            # ===== 内窥镜特定参数 =====
            # 镜面反射：高光强度
            'endo_specular_intensity': {1: 0.15, 2: 0.25, 3: 0.40, 4: 0.55, 5: 0.70},
            # 气泡：覆盖面积比例
            'endo_bubble_coverage': {1: 0.02, 2: 0.04, 3: 0.07, 4: 0.10, 5: 0.15},
            # 血液：覆盖面积比例
            'endo_blood_coverage': {1: 0.02, 2: 0.05, 3: 0.10, 4: 0.18, 5: 0.28},
            # 过曝：过曝区域比例
            'endo_saturation_ratio': {1: 0.01, 2: 0.03, 3: 0.06, 4: 0.10, 5: 0.15},

            # ===== OCT特定参数 =====
            # 阴影：垂直信号丢失的宽度比例
            'oct_shadow_width': {1: 0.03, 2: 0.05, 3: 0.08, 4: 0.12, 5: 0.18},
            # 眨眼/运动：水平条纹数量
            'oct_blink_count': {1: 1, 2: 2, 3: 3, 4: 4, 5: 6},
            # 运动错位：错位幅度（像素比例）
            'oct_motion_shift': {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.10, 5: 0.15},
            # 散斑噪声强度
            'oct_speckle_intensity': {1: 0.08, 2: 0.15, 3: 0.25, 4: 0.35, 5: 0.50},

            # ===== 病理切片特定参数 =====
			'path_stain_intensity': {1: 0.08, 2: 0.12, 3: 0.18, 4: 0.25, 5: 0.35},
			'path_bubble_count': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
			'path_bubble_cracks': {1: 0, 2: 2, 3: 4, 4: 6, 5: 8},
        }
    
    def _apply_level_multipliers(self):
        """
        应用level强度乘数到所有扰动参数
        
        说明：
        - 对于普通数值参数：直接乘以乘数
        - 对于tuple参数（如brightness_contrast）：每个元素都乘以乘数
        - 对于compression_artifacts：质量越低表示扰动越强，所以反向计算
        - 对于pixelate/scale：数值越小表示扰动越强，也反向计算
        """
        for param_name, level_params in self.severity_params.items():
            for level in level_params.keys():
                multiplier = self.level_multipliers.get(level, 1.0)
                
                if multiplier == 1.0:
                    continue  # 不需要修改
                
                original_value = level_params[level]
                
                # 根据参数类型应用乘数
                if isinstance(original_value, tuple):
                    # brightness_contrast类型：(brightness, contrast)
                    level_params[level] = tuple(v * multiplier for v in original_value)
                
                elif param_name == 'compression_artifacts':
                    # 压缩质量：质量越低扰动越强
                    # 公式：new_quality = 100 - (100 - original_quality) * multiplier
                    degradation = 100 - original_value
                    new_quality = 100 - degradation * multiplier
                    level_params[level] = max(1, min(100, int(new_quality)))
                
                elif param_name in ['pixelate', 'scale']:
                    # 这些参数数值越小扰动越强
                    # 公式：new_value = 1 - (1 - original_value) * multiplier
                    degradation = 1 - original_value
                    new_value = 1 - degradation * multiplier
                    level_params[level] = max(0.1, min(1.0, new_value))
                
                elif isinstance(original_value, int):
                    # 整数类型（如motion_blur kernel size, mri_ghost_count等）
                    level_params[level] = max(1, int(original_value * multiplier))
                
                else:
                    # 其他float类型参数（大部分扰动）
                    level_params[level] = original_value * multiplier
    
    def detect_modality_from_path(self, image_path: str) -> str:
        path_lower = image_path.lower()
        for keyword, modality in self.dataset_to_modality.items():
            if keyword in path_lower:
                return modality
        return 'PATHOLOGY'
    
    def _is_color_image(self, image: np.ndarray) -> bool:
        return image.ndim == 3 and image.shape[2] in [3, 4]
    
    def load_image(self, file_path: str, is_npy: bool = False, npy_format: str = 'single') -> np.ndarray:
        try:
            if is_npy:
                array = np.load(file_path)
                if npy_format == 'multi' and array.ndim > 2:
                    array = array[..., 0]
                if array.ndim > 2 and array.shape[0] < 10:
                    array = array[array.shape[0]//2]
                return array.astype(np.float32)
            else:
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        raise ValueError(f"Cannot read image: {file_path}")
                    if image.ndim == 3:
                        if image.shape[2] == 3:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        elif image.shape[2] == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    return image.astype(np.float32)
                else:
                    image = sitk.ReadImage(file_path)
                    array = sitk.GetArrayFromImage(image)
                    if array.ndim > 2:
                        array = array[array.shape[0]//2]
                    return array.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Failed to load image {file_path}: {str(e)}")
    
    def save_image(self, image: np.ndarray, file_path: str):
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        
        if image.dtype != np.uint8:
            if self._is_color_image(image):
                img_normalized = np.zeros_like(image, dtype=np.uint8)
                for c in range(image.shape[2]):
                    channel = image[:, :, c]
                    img_normalized[:, :, c] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img_normalized = image.copy()
        
        if self._is_color_image(img_normalized):
            if img_normalized.shape[2] == 3:
                img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2BGR)
            elif img_normalized.shape[2] == 4:
                img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_RGBA2BGRA)
        
        cv2.imwrite(file_path, img_normalized)
    
    def compute_ssim(self, original: np.ndarray, perturbed: np.ndarray) -> float:
        orig_norm = self._normalize_for_metrics(original)
        pert_norm = self._normalize_for_metrics(perturbed)
        
        if self._is_color_image(original):
            ssim_values = []
            for c in range(original.shape[2]):
                ssim_val = ssim(orig_norm[:, :, c], pert_norm[:, :, c], data_range=1.0)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            return ssim(orig_norm, pert_norm, data_range=1.0)
    
    def compute_mse(self, original: np.ndarray, perturbed: np.ndarray) -> float:
        orig_norm = self._normalize_for_metrics(original)
        pert_norm = self._normalize_for_metrics(perturbed)
        return np.mean((orig_norm - pert_norm) ** 2)
    
    def _normalize_for_metrics(self, image: np.ndarray) -> np.ndarray:
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            return (image - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(image)
    
    def _get_param_range(self, perturbation_type: str) -> Tuple[float, float]:
        param_ranges = {
            'gaussian_noise': (0.001, 0.3),
            'salt_pepper_noise': (0.0001, 0.1),
            'speckle_noise': (0.01, 0.8),
            'gaussian_blur': (0.1, 10.0),
            'motion_blur': (3, 25),
            'brightness_contrast': (0.01, 0.6),
            'compression_artifacts': (5, 95),
            'pixelate': (0.1, 0.95),
            'rotation': (1, 45),
            'scale': (0.5, 0.98),
            'translation': (1, 50),
            'motion_artifacts': (0.5, 50.0),
            'bias_field': (0.1, 0.98),
            'ghosting': (0.05, 1.0),
            'dermoscopy_artifacts': (0.05, 0.8),
            'light_reflection': (0.1, 0.8),
            'hair_artifacts': (1, 20),
            'acoustic_shadowing': (0.1, 0.95),  # us_shadow_attenuation
            'reverberation_artifacts': (0.05, 0.6),  # us_reverb_intensity
            'specular_reflection': (0.1, 0.85),      # endo_specular_intensity
            'bubbles': (0.01, 0.20),                 # endo_bubble_coverage
            'blood': (0.01, 0.35),                   # endo_blood_coverage
            'saturation': (0.005, 0.20),             # endo_saturation_ratio
            
            'shadow_artifacts': (0.02, 0.25),
            'blink_artifacts': (1, 8),   
            'motion_artifacts_oct': (0.01, 0.20), 
            'defocus_artifacts': (0.3, 4.0),
            'speckle_artifacts': (0.05, 0.60), 
            'motion_lines': (1, 15),
            'banding_artifacts': (1, 10),
            'stain_variation': (0.05, 0.8),
            'bubble_artifacts': (1, 15),
            'metal_artifacts': (0.05, 0.5),
            'beam_hardening': (0.05, 0.5),
            'photon_starvation': (0.01, 0.4),
        }
        return param_ranges.get(perturbation_type, (0.01, 1.0))
    
    def _is_inverse_param(self, perturbation_type: str) -> bool:
        inverse_params = ['compression_artifacts', 'pixelate', 'scale']
        return perturbation_type in inverse_params
    
    def _apply_perturbation_with_param(self, image: np.ndarray, modality: str,
                                       perturbation_type: str, param_value: float) -> np.ndarray:
        original_params = {}
        
        if perturbation_type == 'gaussian_noise':
            original_params['gaussian_noise'] = self.severity_params['gaussian_noise'].copy()
            for s in range(1, 6):
                self.severity_params['gaussian_noise'][s] = param_value
            result = self._gaussian_noise(image, 3, modality)
            
        elif perturbation_type == 'salt_pepper_noise':
            original_params['salt_pepper_noise'] = self.severity_params['salt_pepper_noise'].copy()
            for s in range(1, 6):
                self.severity_params['salt_pepper_noise'][s] = param_value
            result = self._salt_pepper_noise(image, 3, modality)
            
        elif perturbation_type == 'speckle_noise':
            original_params['speckle_noise'] = self.severity_params['speckle_noise'].copy()
            for s in range(1, 6):
                self.severity_params['speckle_noise'][s] = param_value
            result = self._speckle_noise(image, 3, modality)
            
        elif perturbation_type == 'gaussian_blur':
            original_params['gaussian_blur'] = self.severity_params['gaussian_blur'].copy()
            for s in range(1, 6):
                self.severity_params['gaussian_blur'][s] = param_value
            result = self._gaussian_blur(image, 3, modality)
            
        elif perturbation_type == 'motion_blur':
            original_params['motion_blur'] = self.severity_params['motion_blur'].copy()
            for s in range(1, 6):
                self.severity_params['motion_blur'][s] = int(param_value)
            result = self._motion_blur(image, 3, modality)
            
        elif perturbation_type == 'brightness_contrast':
            original_params['brightness_contrast'] = self.severity_params['brightness_contrast'].copy()
            for s in range(1, 6):
                self.severity_params['brightness_contrast'][s] = (param_value, param_value)
            result = self._brightness_contrast(image, 3, modality)
            
        elif perturbation_type == 'compression_artifacts':
            original_params['compression_artifacts'] = self.severity_params['compression_artifacts'].copy()
            for s in range(1, 6):
                self.severity_params['compression_artifacts'][s] = int(param_value)
            result = self._compression_artifacts(image, 3, modality)
            
        elif perturbation_type == 'pixelate':
            original_params['pixelate'] = self.severity_params['pixelate'].copy()
            for s in range(1, 6):
                self.severity_params['pixelate'][s] = param_value
            result = self._pixelate(image, 3, modality)
            
        elif perturbation_type == 'rotation':
            original_params['rotation'] = self.severity_params['rotation'].copy()
            for s in range(1, 6):
                self.severity_params['rotation'][s] = param_value
            result = self._rotation(image, 3, modality)
            
        elif perturbation_type == 'scale':
            original_params['scale'] = self.severity_params['scale'].copy()
            for s in range(1, 6):
                self.severity_params['scale'][s] = param_value
            result = self._scale(image, 3, modality)
            
        elif perturbation_type == 'translation':
            original_params['translation'] = self.severity_params['translation'].copy()
            for s in range(1, 6):
                self.severity_params['translation'][s] = int(param_value)
            result = self._translation(image, 3, modality)
            
        elif perturbation_type == 'motion_artifacts':
            original_params['mri_motion_translation'] = self.severity_params['mri_motion_translation'].copy()
            for s in range(1, 6):
                self.severity_params['mri_motion_translation'][s] = param_value
            result = self._mri_motion_artifacts(image, 3)
            
        elif perturbation_type == 'bias_field':
            original_params['mri_bias_intensity'] = self.severity_params['mri_bias_intensity'].copy()
            for s in range(1, 6):
                self.severity_params['mri_bias_intensity'][s] = param_value
            result = self._mri_bias_field(image, 3)
            
        elif perturbation_type == 'ghosting':
            original_params['mri_ghost_intensity'] = self.severity_params['mri_ghost_intensity'].copy()
            for s in range(1, 6):
                self.severity_params['mri_ghost_intensity'][s] = param_value
            result = self._mri_ghosting(image, 3)
        
        # 超声特定扰动
        elif perturbation_type == 'acoustic_shadowing':
            original_params['us_shadow_attenuation'] = self.severity_params['us_shadow_attenuation'].copy()
            for s in range(1, 6):
                self.severity_params['us_shadow_attenuation'][s] = param_value
            result = self._us_acoustic_shadowing(image, 3)
            
        elif perturbation_type == 'reverberation_artifacts':
            original_params['us_reverb_intensity'] = self.severity_params['us_reverb_intensity'].copy()
            for s in range(1, 6):
                self.severity_params['us_reverb_intensity'][s] = param_value
            result = self._us_reverberation(image, 3)
        
        # 内窥镜特定扰动
        elif perturbation_type == 'specular_reflection':
            original_params['endo_specular_intensity'] = self.severity_params['endo_specular_intensity'].copy()
            for s in range(1, 6):
                self.severity_params['endo_specular_intensity'][s] = param_value
            result = self._endo_specular_reflection(image, 3)
            
        elif perturbation_type == 'bubbles':
            original_params['endo_bubble_coverage'] = self.severity_params['endo_bubble_coverage'].copy()
            for s in range(1, 6):
                self.severity_params['endo_bubble_coverage'][s] = param_value
            result = self._endo_bubbles(image, 3)
            
        elif perturbation_type == 'blood':
            original_params['endo_blood_coverage'] = self.severity_params['endo_blood_coverage'].copy()
            for s in range(1, 6):
                self.severity_params['endo_blood_coverage'][s] = param_value
            result = self._endo_blood(image, 3)
            
        elif perturbation_type == 'saturation':
            original_params['endo_saturation_ratio'] = self.severity_params['endo_saturation_ratio'].copy()
            for s in range(1, 6):
                self.severity_params['endo_saturation_ratio'][s] = param_value
            result = self._endo_saturation(image, 3)

        elif perturbation_type == 'shadow_artifacts':
            original_params['oct_shadow_width'] = self.severity_params['oct_shadow_width'].copy()
            for s in range(1, 6):
                self.severity_params['oct_shadow_width'][s] = param_value
            result = self._oct_shadow(image, 3)

        elif perturbation_type == 'blink_artifacts':
            original_params['oct_blink_count'] = self.severity_params['oct_blink_count'].copy()
            for s in range(1, 6):
                self.severity_params['oct_blink_count'][s] = int(param_value)
            result = self._oct_blink(image, 3)
            
        elif perturbation_type == 'motion_artifacts_oct':
            original_params['oct_motion_shift'] = self.severity_params['oct_motion_shift'].copy()
            for s in range(1, 6):
                self.severity_params['oct_motion_shift'][s] = param_value
            result = self._oct_motion(image, 3)
        
        elif perturbation_type == 'defocus_artifacts':
            original_params['gaussian_blur'] = self.severity_params['gaussian_blur'].copy()
            for s in range(1, 6):
                self.severity_params['gaussian_blur'][s] = param_value
            result = self._oct_defocus(image, 3)
            
        elif perturbation_type == 'speckle_artifacts':
            original_params['oct_speckle_intensity'] = self.severity_params['oct_speckle_intensity'].copy()
            for s in range(1, 6):
                self.severity_params['oct_speckle_intensity'][s] = param_value
            result = self._oct_speckle(image, 3)

        # 病理特定扰动
        elif perturbation_type == 'stain_variation':
            original_params['path_stain_intensity'] = self.severity_params['path_stain_intensity'].copy()
            for s in range(1, 6):
                self.severity_params['path_stain_intensity'][s] = param_value
            result = self._path_stain_variation(image, 3)
        
        elif perturbation_type == 'bubble_artifacts':
            original_params['path_bubble_count'] = self.severity_params['path_bubble_count'].copy()
            for s in range(1, 6):
                self.severity_params['path_bubble_count'][s] = int(param_value)
            result = self._path_bubble_artifacts(image, 3)
            
        else:
            result = self.apply_perturbation(image, modality, perturbation_type, 3)
        
        for key, value in original_params.items():
            self.severity_params[key] = value
        
        return result
    
    def find_adaptive_param(self, image: np.ndarray, modality: str, perturbation_type: str,
                           target_ssim_min: float, target_ssim_max: float,
                           max_iterations: int = 20, tolerance: float = 0.02) -> Tuple[float, float, np.ndarray]:
        """使用二分搜索找到使SSIM落在目标范围内的参数值"""
        param_min, param_max = self._get_param_range(perturbation_type)
        is_inverse = self._is_inverse_param(perturbation_type)
        
        target_ssim = (target_ssim_min + target_ssim_max) / 2
        
        best_param = None
        best_ssim = None
        best_image = None
        best_diff = float('inf')
        
        for iteration in range(max_iterations):
            param = (param_min + param_max) / 2
            
            try:
                perturbed = self._apply_perturbation_with_param(image, modality, perturbation_type, param)
                current_ssim = self.compute_ssim(image, perturbed)
            except Exception as e:
                if is_inverse:
                    param_min = param
                else:
                    param_max = param
                continue
            
            diff_to_target = abs(current_ssim - target_ssim)
            if diff_to_target < best_diff:
                best_diff = diff_to_target
                best_param = param
                best_ssim = current_ssim
                best_image = perturbed
            
            if target_ssim_min - tolerance <= current_ssim <= target_ssim_max + tolerance:
                return param, current_ssim, perturbed
            
            if is_inverse:
                if current_ssim > target_ssim:
                    param_max = param
                else:
                    param_min = param
            else:
                if current_ssim > target_ssim:
                    param_min = param
                else:
                    param_max = param
        
        if best_image is not None:
            return best_param, best_ssim, best_image
        else:
            perturbed = self.apply_perturbation(image, modality, perturbation_type, 3)
            return param, self.compute_ssim(image, perturbed), perturbed
    
    def apply_perturbation_adaptive(self, image: np.ndarray, modality: str,
                                    perturbation_type: str, severity: int = 3,
                                    ssim_targets: Dict[int, Tuple[float, float]] = None,
                                    use_cache: bool = True) -> Tuple[np.ndarray, float, float]:
        """
        应用扰动并自适应调整参数
        use_cache=True: 第一张图搜索参数，后续复用缓存
        """
        if severity == 0:
        	return image.copy(), 1.0, None
        
        if ssim_targets is None:
            ssim_targets = self.ssim_targets if self.ssim_targets else self.default_ssim_targets
        
        if ssim_targets is None or severity not in ssim_targets:
            perturbed = self.apply_perturbation(image, modality, perturbation_type, severity)
            ssim_val = self.compute_ssim(image, perturbed)
            return perturbed, ssim_val, None
        
        # ========== 参数缓存逻辑 ==========
        if use_cache:
            cached_param = self.get_cached_param(perturbation_type, severity)
            if cached_param is not None:
                # 直接使用缓存参数，无需二分搜索
                perturbed = self._apply_perturbation_with_param(image, modality, perturbation_type, cached_param)
                ssim_val = self.compute_ssim(image, perturbed)
                return perturbed, ssim_val, cached_param
        
        # 第一张图：完整自适应搜索
        target_min, target_max = ssim_targets[severity]
        param, ssim_val, perturbed = self.find_adaptive_param(
            image, modality, perturbation_type, target_min, target_max
        )
        
        # 缓存参数
        if use_cache and param is not None:
            self.set_cached_param(perturbation_type, severity, param)
        
        return perturbed, ssim_val, param
    
    def apply_perturbation(self, image: np.ndarray, modality: str, 
                          perturbation_type: str, severity: int = 3) -> np.ndarray:

        if severity == 0:
        	return image.copy()
    	
        if not 1 <= severity <= 5:
            raise ValueError("Severity must be between 1 and 5")
        
        if perturbation_type in self._get_base_perturbations():
            return self._apply_base_perturbation(image, modality, perturbation_type, severity)
        elif perturbation_type in self._get_modality_perturbations(modality):
            return self._apply_modality_perturbation(image, modality, perturbation_type, severity)
        else:
            raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    def _get_base_perturbations(self) -> List[str]:
        return ['gaussian_noise', 'salt_pepper_noise', 'speckle_noise', 'gaussian_blur', 
                'brightness_contrast', 'compression_artifacts', 'motion_blur', 'pixelate',
                'rotation', 'scale', 'translation']
    
    def _get_modality_perturbations(self, modality: str) -> List[str]:
        modality_perturbations = {
            'DERM': ['dermoscopy_artifacts', 'light_reflection', 'hair_artifacts'],
            'MRI_BRAIN': ['motion_artifacts', 'bias_field', 'ghosting'],
            'US_BREAST': ['acoustic_shadowing', 'reverberation_artifacts'],
            'ENDOSCOPY': ['specular_reflection', 'bubbles', 'blood', 'saturation'],
            # 'OCT': ['shadow_artifacts', 'defocus_artifacts', 'motion_lines', 'banding_artifacts'],
            'OCT': ['shadow_artifacts', 'blink_artifacts', 'motion_artifacts_oct', 'defocus_artifacts', 'speckle_artifacts'],
            'PATHOLOGY': ['stain_variation', 'bubble_artifacts'],
            'CT_ABDOMEN': ['metal_artifacts', 'beam_hardening', 'photon_starvation']
        }
        return modality_perturbations.get(modality, [])
    
    def _apply_base_perturbation(self, image: np.ndarray, modality: str, 
                               perturbation_type: str, severity: int) -> np.ndarray:
        if perturbation_type == 'gaussian_noise':
            return self._gaussian_noise(image, severity, modality)
        elif perturbation_type == 'salt_pepper_noise':
            return self._salt_pepper_noise(image, severity, modality)
        elif perturbation_type == 'speckle_noise':
            return self._speckle_noise(image, severity, modality)
        elif perturbation_type == 'gaussian_blur':
            return self._gaussian_blur(image, severity, modality)
        elif perturbation_type == 'brightness_contrast':
            return self._brightness_contrast(image, severity, modality)
        elif perturbation_type == 'compression_artifacts':
            return self._compression_artifacts(image, severity, modality)
        elif perturbation_type == 'motion_blur':
            return self._motion_blur(image, severity, modality)
        elif perturbation_type == 'pixelate':
            return self._pixelate(image, severity, modality)
        elif perturbation_type == 'rotation':
            return self._rotation(image, severity, modality)
        elif perturbation_type == 'scale':
            return self._scale(image, severity, modality)
        elif perturbation_type == 'translation':
            return self._translation(image, severity, modality)
        else:
            raise ValueError(f"Unknown base perturbation type: {perturbation_type}")
    
    def _gaussian_noise(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        sigma = self.severity_params['gaussian_noise'][severity]
        modality_factor = {'DERM': 0.8, 'MRI_BRAIN': 1.0, 'US_BREAST': 1.5, 'ENDOSCOPY': 0.7,
                          'OCT': 0.9, 'PATHOLOGY': 0.6, 'CT_ABDOMEN': 1.2}.get(modality, 1.0)
        sigma = sigma * modality_factor
        noise = np.random.normal(0, sigma * (image.max() - image.min()), image.shape)
        return np.clip(image + noise, image.min(), image.max())
    
    def _salt_pepper_noise(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        prob = self.severity_params['salt_pepper_noise'][severity]
        noisy = image.copy()
        if self._is_color_image(image):
            h, w = image.shape[:2]
            salt_mask = np.random.random((h, w)) < prob / 2
            pepper_mask = np.random.random((h, w)) < prob / 2
            for c in range(image.shape[2]):
                noisy[:, :, c][salt_mask] = image[:, :, c].max()
                noisy[:, :, c][pepper_mask] = image[:, :, c].min()
        else:
            salt = np.random.random(image.shape) < prob / 2
            pepper = np.random.random(image.shape) < prob / 2
            noisy[salt] = image.max()
            noisy[pepper] = image.min()
        return noisy
    
    def _speckle_noise(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        sigma = self.severity_params['speckle_noise'][severity]
        noise = np.random.normal(1.0, sigma, image.shape)
        noisy = image * noise
        return np.clip(noisy, image.min(), image.max())
    
    def _gaussian_blur(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        sigma = self.severity_params['gaussian_blur'][severity]
        if self._is_color_image(image):
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = gaussian_filter(image[:, :, c], sigma=sigma)
            return blurred
        else:
            return gaussian_filter(image, sigma=sigma)
    
    def _motion_blur(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        kernel_size = self.severity_params['motion_blur'][severity]
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size//2, :] = 1.0
        kernel = kernel / kernel_size
        angle = np.random.uniform(-180, 180)
        M = cv2.getRotationMatrix2D((kernel_size//2, kernel_size//2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        if self._is_color_image(image):
            blurred = np.zeros_like(image)
            for c in range(image.shape[2]):
                blurred[:, :, c] = cv2.filter2D(image[:, :, c].astype(np.float32), -1, kernel)
            return blurred
        else:
            return cv2.filter2D(image.astype(np.float32), -1, kernel)
    
    def _brightness_contrast(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        brightness_delta, contrast_delta = self.severity_params['brightness_contrast'][severity]
        brightness_factor = 1.0 + np.random.uniform(-brightness_delta, brightness_delta)
        contrast_factor = 1.0 + np.random.uniform(-contrast_delta, contrast_delta)
        if self._is_color_image(image):
            adjusted = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                mean = channel.mean()
                adj_channel = (channel - mean) * contrast_factor + mean
                adj_channel = adj_channel * brightness_factor
                adjusted[:, :, c] = np.clip(adj_channel, channel.min(), channel.max())
            return adjusted
        else:
            mean = image.mean()
            adjusted = (image - mean) * contrast_factor + mean
            adjusted = adjusted * brightness_factor
            return np.clip(adjusted, image.min(), image.max())
    
    def _compression_artifacts(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        quality = self.severity_params['compression_artifacts'][severity]
        if self._is_color_image(image):
            img_uint8 = np.zeros_like(image, dtype=np.uint8)
            for c in range(image.shape[2]):
                img_uint8[:, :, c] = cv2.normalize(image[:, :, c], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
            decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
            decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
            return decimg.astype(np.float32)
        else:
            img_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encimg = cv2.imencode('.jpg', img_uint8, encode_param)
            decimg = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
            return decimg.astype(np.float32)
    
    def _pixelate(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        scale_factor = self.severity_params['pixelate'][severity]
        h, w = image.shape[:2]
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return pixelated
    
    def _rotation(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        angle = self.severity_params['rotation'][severity] * np.random.choice([-1, 1])
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def _scale(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        scale_factor = self.severity_params['scale'][severity]
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(image, (new_w, new_h))
        result = np.zeros_like(image)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
        return result
    
    def _translation(self, image: np.ndarray, severity: int, modality: str) -> np.ndarray:
        shift = self.severity_params['translation'][severity]
        dx = np.random.randint(-shift, shift)
        dy = np.random.randint(-shift, shift)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)
        return translated
    
    def _apply_modality_perturbation(self, image: np.ndarray, modality: str, 
                                   perturbation_type: str, severity: int) -> np.ndarray:
        if modality == 'MRI_BRAIN':
            if perturbation_type == 'motion_artifacts':
                return self._mri_motion_artifacts(image, severity)
            elif perturbation_type == 'bias_field':
                return self._mri_bias_field(image, severity)
            elif perturbation_type == 'ghosting':
                return self._mri_ghosting(image, severity)
        elif modality == 'DERM':
            if perturbation_type == 'dermoscopy_artifacts':
                return self._derm_artifacts(image, severity)
            elif perturbation_type == 'light_reflection':
                return self._derm_light_reflection(image, severity)
            elif perturbation_type == 'hair_artifacts':
                return self._derm_hair_artifacts(image, severity)
        elif modality == 'US_BREAST':
            if perturbation_type == 'acoustic_shadowing':
                return self._us_acoustic_shadowing(image, severity)
            elif perturbation_type == 'reverberation_artifacts':
                return self._us_reverberation(image, severity)
        elif modality == 'ENDOSCOPY':
            if perturbation_type == 'specular_reflection':
                return self._endo_specular_reflection(image, severity)
            elif perturbation_type == 'bubbles':
                return self._endo_bubbles(image, severity)
            elif perturbation_type == 'blood':
                return self._endo_blood(image, severity)
            elif perturbation_type == 'saturation':
                return self._endo_saturation(image, severity)
        elif modality == 'OCT':
            if perturbation_type == 'shadow_artifacts':
                return self._oct_shadow(image, severity)
            elif perturbation_type == 'blink_artifacts':
                return self._oct_blink(image, severity)
            elif perturbation_type == 'motion_artifacts_oct':
                return self._oct_motion(image, severity)
            elif perturbation_type == 'defocus_artifacts':
                return self._oct_defocus(image, severity)
            elif perturbation_type == 'speckle_artifacts':
                return self._oct_speckle(image, severity)
        elif modality == 'PATHOLOGY':
            if perturbation_type == 'stain_variation':
                return self._path_stain_variation(image, severity)
            elif perturbation_type == 'bubble_artifacts':
                return self._path_bubble_artifacts(image, severity)
        elif modality == 'CT_ABDOMEN':
            if perturbation_type == 'metal_artifacts':
                return self._ct_metal_artifacts(image, severity)
            elif perturbation_type == 'beam_hardening':
                return self._ct_beam_hardening(image, severity)
            elif perturbation_type == 'photon_starvation':
                return self._ct_photon_starvation(image, severity)
        raise ValueError(f"Unknown modality-specific perturbation: {perturbation_type} for {modality}")
    
    # === MRI Brain Perturbations ===
    def _mri_motion_artifacts(self, image: np.ndarray, severity: int) -> np.ndarray:
        if self._is_color_image(image):
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._mri_motion_artifacts_single(image[:, :, c], severity)
            return result
        else:
            return self._mri_motion_artifacts_single(image, severity)
    
    def _mri_motion_artifacts_single(self, image: np.ndarray, severity: int) -> np.ndarray:
        h, w = image.shape
        k_space = np.fft.fft2(image)
        k_space = np.fft.fftshift(k_space)
        max_translation = self.severity_params['mri_motion_translation'][severity]
        corrupted_lines_pct = self.severity_params['mri_motion_corrupted_pct'][severity]
        num_corrupted_lines = int(h * corrupted_lines_pct)
        center_protection = max(h // 20, 2)
        center_start = h // 2 - center_protection // 2
        center_end = h // 2 + center_protection // 2
        available_lines = list(range(0, center_start)) + list(range(center_end, h))
        rng = np.random.RandomState(self.seed + severity * 100)
        corrupted_lines = rng.choice(available_lines, size=min(num_corrupted_lines, len(available_lines)), replace=False)
        translation_x = np.zeros(h)
        for line in corrupted_lines:
            trans = rng.uniform(-max_translation, max_translation)
            translation_x[line] = trans
        translation_x = gaussian_filter(translation_x, sigma=0.5)
        corrupted_k = k_space.copy().astype(np.complex128)
        freq_x = np.fft.fftfreq(w)
        for y in range(h):
            if abs(translation_x[y]) > 0.01:
                phase_ramp = np.exp(-2j * np.pi * freq_x * translation_x[y])
                corrupted_k[y, :] *= phase_ramp
        corrupted_k = np.fft.ifftshift(corrupted_k)
        corrupted = np.abs(np.fft.ifft2(corrupted_k))
        img_min, img_max = image.min(), image.max()
        corr_min, corr_max = corrupted.min(), corrupted.max()
        if corr_max - corr_min > 1e-8:
            corrupted = (corrupted - corr_min) / (corr_max - corr_min) * (img_max - img_min) + img_min
        return corrupted.astype(np.float32)
    
    def _mri_bias_field(self, image: np.ndarray, severity: int) -> np.ndarray:
        h, w = image.shape[:2]
        intensity = self.severity_params['mri_bias_intensity'][severity]
        low_res_size = 2 + severity * 3
        rng = np.random.RandomState(self.seed + severity * 200)
        low_res_field = rng.rand(low_res_size, low_res_size)
        bias_field = cv2.resize(low_res_field, (w, h), interpolation=cv2.INTER_CUBIC)
        sigma = min(h, w) / (15 + severity * 5)
        bias_field = gaussian_filter(bias_field, sigma=sigma)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        for freq_mult in range(1, severity + 2):
            freq = freq_mult * np.pi / max(h, w) * 3
            phase = rng.uniform(0, 2 * np.pi)
            modulation = 0.2 * np.sin(freq * x_coords + phase) * np.cos(freq * y_coords * 0.8 + phase)
            bias_field = bias_field + modulation * intensity
        center_y, center_x = h / 2, w / 2
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        radial_falloff = 1.0 - (dist / max_dist) * intensity * 0.5
        bias_field = bias_field * radial_falloff
        bias_min, bias_max = bias_field.min(), bias_field.max()
        if bias_max - bias_min > 1e-8:
            bias_field = (bias_field - bias_min) / (bias_max - bias_min)
            bias_field = (1.0 - intensity) + bias_field * (2 * intensity)
        else:
            bias_field = np.ones((h, w))
        if self._is_color_image(image):
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = image[:, :, c] * bias_field
                result[:, :, c] = np.clip(result[:, :, c], image[:, :, c].min(), image[:, :, c].max())
            return result
        else:
            result = image * bias_field
            return np.clip(result, image.min(), image.max()).astype(np.float32)
    
    def _mri_ghosting(self, image: np.ndarray, severity: int) -> np.ndarray:
        ghost_intensity = self.severity_params['mri_ghost_intensity'][severity]
        num_ghosts = self.severity_params['mri_ghost_count'][severity]
        if self._is_color_image(image):
            result = image.copy()
            for c in range(image.shape[2]):
                result[:, :, c] = self._mri_ghosting_single(image[:, :, c], severity, ghost_intensity, num_ghosts)
            return result
        else:
            return self._mri_ghosting_single(image, severity, ghost_intensity, num_ghosts)
    
    def _mri_ghosting_single(self, image: np.ndarray, severity: int, ghost_intensity: float, num_ghosts: int) -> np.ndarray:
        h, w = image.shape
        result = image.copy().astype(np.float64)
        for i in range(1, num_ghosts + 1):
            offset_y = (h // 2) * i % h
            decay = 1.0 / (i ** 0.8)
            current_intensity = ghost_intensity * decay
            ghost = np.roll(image, offset_y, axis=0)
            ghost = gaussian_filter(ghost, sigma=0.2 + 0.1 * i)
            result = result + ghost * current_intensity
            if severity >= 3:
                offset_x = (w // 4) * i % w
                ghost_x = np.roll(image, offset_x, axis=1)
                ghost_x = gaussian_filter(ghost_x, sigma=0.3 + 0.1 * i)
                result = result + ghost_x * current_intensity * 0.3
        img_min, img_max = image.min(), image.max()
        result_min, result_max = result.min(), result.max()
        if result_max - result_min > 1e-8:
            result = img_min + (result - result_min) * (img_max - img_min) / (result_max - result_min)
        result = np.clip(result, img_min, img_max)
        return result.astype(np.float32)
    
    # === Dermatology Perturbations ===
    def _derm_artifacts(self, image: np.ndarray, severity: int) -> np.ndarray:
        h, w = image.shape[:2]
        artifact = image.copy()
        radius = min(h, w) // 2
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        vignette_intensity = severity * 0.1
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        vignette = 1.0 - vignette_intensity * (distance / radius)
        vignette = np.clip(vignette, 0, 1)
        if self._is_color_image(image):
            for c in range(image.shape[2]):
                artifact[:, :, c] = artifact[:, :, c] * vignette
        else:
            artifact = artifact * vignette
        return artifact
    
    def _derm_light_reflection(self, image: np.ndarray, severity: int) -> np.ndarray:
        h, w = image.shape[:2]
        artifact = image.copy()
        center_y = np.random.randint(h // 4, 3 * h // 4)
        center_x = np.random.randint(w // 4, 3 * w // 4)
        sigma = min(h, w) / (10 - severity)
        y, x = np.ogrid[:h, :w]
        reflection = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        intensity = severity * 0.3
        if self._is_color_image(image):
            for c in range(image.shape[2]):
                artifact[:, :, c] = artifact[:, :, c] + reflection * intensity * image[:, :, c].max()
                artifact[:, :, c] = np.clip(artifact[:, :, c], image[:, :, c].min(), image[:, :, c].max())
        else:
            artifact = artifact + reflection * intensity * image.max()
            artifact = np.clip(artifact, image.min(), image.max())
        return artifact
    
    def _derm_hair_artifacts(self, image: np.ndarray, severity: int) -> np.ndarray:
        h, w = image.shape[:2]
        artifact = image.copy()
        num_hairs = severity * 3
        hair_mask = np.zeros((h, w), dtype=np.uint8)
        for _ in range(num_hairs):
            x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
            main_angle = np.random.uniform(0, np.pi)
            length = np.random.randint(50, min(h, w) // 2)
            x3 = int(x0 + length * np.cos(main_angle))
            y3 = int(y0 + length * np.sin(main_angle))
            curvature = np.random.uniform(0.1, 0.4) * length
            curve_dir = np.random.choice([-1, 1])
            x1 = int(x0 + length * 0.33 * np.cos(main_angle) + curvature * curve_dir * np.cos(main_angle + np.pi/2))
            y1 = int(y0 + length * 0.33 * np.sin(main_angle) + curvature * curve_dir * np.sin(main_angle + np.pi/2))
            curve_dir2 = curve_dir * np.random.choice([1, -1], p=[0.7, 0.3])
            x2 = int(x0 + length * 0.67 * np.cos(main_angle) + curvature * 0.5 * curve_dir2 * np.cos(main_angle + np.pi/2))
            y2 = int(y0 + length * 0.67 * np.sin(main_angle) + curvature * 0.5 * curve_dir2 * np.sin(main_angle + np.pi/2))
            points = []
            for t in np.linspace(0, 1, 100):
                x = int((1-t)**3 * x0 + 3*(1-t)**2*t * x1 + 3*(1-t)*t**2 * x2 + t**3 * x3)
                y = int((1-t)**3 * y0 + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y3)
                points.append((x, y))
            base_thickness = max(1, severity // 2 + 1)
            for i in range(len(points) - 1):
                t = i / len(points)
                thickness_factor = 1.0 - 0.5 * abs(t - 0.5) * 2
                thickness = max(1, int(base_thickness * thickness_factor))
                pt1, pt2 = points[i], points[i+1]
                if 0 <= pt1[0] < w and 0 <= pt1[1] < h and 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                    cv2.line(hair_mask, pt1, pt2, 255, thickness)
        hair_mask = cv2.GaussianBlur(hair_mask, (3, 3), 0)
        hair_alpha = hair_mask.astype(np.float32) / 255.0
        hair_value = image.min() * 0.2
        if self._is_color_image(image):
            for c in range(image.shape[2]):
                artifact[:, :, c] = artifact[:, :, c] * (1 - hair_alpha) + hair_value * hair_alpha
        else:
            artifact = artifact * (1 - hair_alpha) + hair_value * hair_alpha
        return artifact
    
    # === Ultrasound Perturbations ===
    def _us_acoustic_shadowing(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        超声声学阴影 - 高声阻抗物体（骨骼、结石）后方的信号衰减区域
        
        物理特性：
        - 阴影从遮挡物下方开始，向深处（图像下方）延伸
        - 越深处阴影越暗（声能持续衰减）
        - 阴影边缘有轻微扩散（声波衍射效应）
        """
        h, w = image.shape[:2]
        artifact = image.copy()
        
        # 从severity_params获取衰减强度（支持自适应模式）
        max_attenuation = self.severity_params['us_shadow_attenuation'][severity]
        
        # 遮挡物位置（阴影起始点）
        rng = np.random.RandomState(self.seed + severity * 300)
        start_y = rng.randint(h // 4, h // 2)
        center_x = rng.randint(w // 4, 3 * w // 4)
        
        # 阴影基础宽度（与attenuation成正相关）
        base_width = int(w * (0.08 + max_attenuation * 0.15))
        
        # 阴影深度
        shadow_depth = h - start_y
        
        for y in range(start_y, h):
            # 深度比例 (0到1)
            depth_ratio = (y - start_y) / shadow_depth
            
            # 阴影宽度随深度略微扩展（声波衍射）
            current_width = int(base_width * (1 + depth_ratio * 0.3))
            left = max(0, center_x - current_width // 2)
            right = min(w, center_x + current_width // 2)
            
            # 核心物理：越深越暗（指数衰减更真实）
            # shadow_factor 从接近1（浅处）降到 (1-max_attenuation)（深处）
            shadow_factor = 1.0 - max_attenuation * (1 - np.exp(-3 * depth_ratio))
            
            # 边缘软化（高斯过渡）
            x_coords = np.arange(w)
            edge_falloff = np.ones(w)
            edge_width = current_width // 6 + 1
            
            # 左边缘过渡
            left_transition = np.clip((x_coords - (left - edge_width)) / edge_width, 0, 1)
            # 右边缘过渡  
            right_transition = np.clip(((right + edge_width) - x_coords) / edge_width, 0, 1)
            edge_falloff = left_transition * right_transition
            
            # 组合：中心区域完全应用shadow_factor，边缘平滑过渡
            row_factor = 1.0 - (1.0 - shadow_factor) * edge_falloff
            
            if self._is_color_image(image):
                for c in range(image.shape[2]):
                    artifact[y, :, c] *= row_factor
            else:
                artifact[y, :] *= row_factor
        
        return artifact
    
    def _us_reverberation(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        超声混响伪影 - 声波在高反射界面间多次往返产生的等间距亮线
        
        物理特性：
        - 等间距的水平亮带（间距 = 2倍第一反射深度）
        - 强度随反射次数指数衰减（每次损失30-50%能量）
        - 混响带有一定厚度，不是单像素线
        - 通常局限于某个水平区域，不是整个图像宽度
        """
        h, w = image.shape[:2]
        artifact = image.copy()
        
        # 从severity_params获取混响强度（支持自适应模式）
        base_intensity = self.severity_params['us_reverb_intensity'][severity]
        
        # 使用固定种子保证可重复性
        rng = np.random.RandomState(self.seed + severity * 400)
        
        # 第一反射界面位置
        interface_y = rng.randint(h // 6, h // 3)
        
        # 混响区域的水平范围（不是整个宽度）
        region_width = w // 2 + rng.randint(0, w // 4)
        region_center = rng.randint(w // 4, 3 * w // 4)
        region_left = max(0, region_center - region_width // 2)
        region_right = min(w, region_center + region_width // 2)
        
        # 混响间距（等于第一反射深度的2倍，这里简化为与深度成比例）
        spacing = int(interface_y * 0.8) + int(base_intensity * 20)
        
        # 混响次数（与强度相关）
        num_echoes = max(3, int(3 + base_intensity * 10))
        
        # 每次混响的厚度（像素）
        band_thickness = max(2, int(3 + base_intensity * 8))
        
        # 能量衰减系数（每次反射保留的能量比例）
        decay_factor = 0.6 - base_intensity * 0.15
        
        for i in range(1, num_echoes + 1):
            echo_center_y = interface_y + i * spacing
            
            if echo_center_y >= h - band_thickness:
                break
            
            # 指数衰减强度
            echo_intensity = base_intensity * (decay_factor ** (i - 1))
            
            # 混响带的y范围
            y_start = max(0, echo_center_y - band_thickness // 2)
            y_end = min(h, echo_center_y + band_thickness // 2 + 1)
            
            for y in range(y_start, y_end):
                # 带内强度分布（中心最强，边缘渐弱）
                dist_from_center = abs(y - echo_center_y)
                band_profile = np.exp(-0.5 * (dist_from_center / (band_thickness / 3)) ** 2)
                local_intensity = echo_intensity * band_profile
                
                # 水平方向的边缘软化
                x_coords = np.arange(w)
                h_profile = np.zeros(w)
                edge_width = (region_right - region_left) // 10 + 1
                
                # 在混响区域内有效，边缘平滑过渡
                mask = (x_coords >= region_left) & (x_coords < region_right)
                h_profile[mask] = 1.0
                
                # 左右边缘高斯过渡
                left_edge = (x_coords >= region_left - edge_width) & (x_coords < region_left)
                right_edge = (x_coords >= region_right) & (x_coords < region_right + edge_width)
                h_profile[left_edge] = np.exp(-0.5 * ((region_left - x_coords[left_edge]) / (edge_width / 2)) ** 2)
                h_profile[right_edge] = np.exp(-0.5 * ((x_coords[right_edge] - region_right) / (edge_width / 2)) ** 2)
                
                # 添加混响（基于原始界面亮度）
                if self._is_color_image(image):
                    for c in range(image.shape[2]):
                        # 混响亮度基于原界面的平均亮度
                        interface_brightness = np.mean(artifact[interface_y, region_left:region_right, c])
                        addition = local_intensity * interface_brightness * h_profile
                        artifact[y, :, c] = np.minimum(
                            artifact[y, :, c] + addition,
                            image[:, :, c].max()
                        )
                else:
                    interface_brightness = np.mean(artifact[interface_y, region_left:region_right])
                    addition = local_intensity * interface_brightness * h_profile
                    artifact[y, :] = np.minimum(
                        artifact[y, :] + addition,
                        image.max()
                    )
        
        return artifact
    
    # === Endoscopy Perturbations ===
    # === Endoscopy Perturbations ===
    
    def _endo_specular_reflection(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        内窥镜镜面反射 - 光源在湿润黏膜表面形成的高光点
        
        物理特性：
        - 高光点通常较小且边缘锐利
        - 中心最亮，快速衰减
        - 多个高光点随机分布
        """
        h, w = image.shape[:2]
        artifact = image.copy()
        
        # 从severity_params获取强度
        intensity = self.severity_params['endo_specular_intensity'][severity]
        
        rng = np.random.RandomState(self.seed + severity * 500)
        
        # 高光点数量与强度相关
        num_highlights = max(1, int(2 + intensity * 8))
        
        for _ in range(num_highlights):
            # 高光位置（偏向图像中心区域）
            center_y = rng.randint(h // 6, 5 * h // 6)
            center_x = rng.randint(w // 6, 5 * w // 6)
            
            # 高光大小（较小，模拟点光源反射）
            sigma = min(h, w) * (0.02 + intensity * 0.03)
            
            y, x = np.ogrid[:h, :w]
            dist_sq = (x - center_x)**2 + (y - center_y)**2
            
            # 使用更锐利的衰减（指数衰减而非高斯）
            highlight = np.exp(-dist_sq / (2 * sigma**2))
            # 增加锐利度
            highlight = highlight ** 0.5
            
            # 高光强度
            highlight_strength = intensity * (0.8 + rng.uniform(0, 0.4))
            
            if self._is_color_image(image):
                for c in range(image.shape[2]):
                    max_val = image[:, :, c].max()
                    artifact[:, :, c] = artifact[:, :, c] + highlight * highlight_strength * max_val
            else:
                artifact = artifact + highlight * highlight_strength * image.max()
        
        # 裁剪到有效范围
        if self._is_color_image(image):
            for c in range(image.shape[2]):
                artifact[:, :, c] = np.clip(artifact[:, :, c], image[:, :, c].min(), image[:, :, c].max())
        else:
            artifact = np.clip(artifact, image.min(), image.max())
        
        return artifact
    
    # def _endo_bubbles(self, image: np.ndarray, severity: int) -> np.ndarray:
    #     """
    #     内窥镜气泡 - 消化道内的气泡或黏液泡
        
    #     基于文献的真实物理特性：
    #     - 气泡内部组织仍可见，但有光学畸变（透镜效果）
    #     - 边缘有局部高光点（镜面反射），不是均匀的环
    #     - 整体有凸起感（protrusion）
    #     """
    #     h, w = image.shape[:2]
    #     artifact = image.copy().astype(np.float32)
        
    #     coverage = self.severity_params['endo_bubble_coverage'][severity]
    #     rng = np.random.RandomState(self.seed + severity * 600)
        
    #     num_bubbles = max(1, int(coverage * 50))
        
    #     for _ in range(num_bubbles):
    #         center_y = rng.randint(h // 8, 7 * h // 8)
    #         center_x = rng.randint(w // 8, 7 * w // 8)
            
    #         radius = rng.randint(max(5, int(min(h, w) * 0.02)), 
    #                              max(10, int(min(h, w) * 0.07)))
            
    #         # 确保气泡在图像范围内
    #         y_min = max(0, center_y - radius - 2)
    #         y_max = min(h, center_y + radius + 3)
    #         x_min = max(0, center_x - radius - 2)
    #         x_max = min(w, center_x + radius + 3)
            
    #         if y_max <= y_min or x_max <= x_min:
    #             continue
            
    #         local_h = y_max - y_min
    #         local_w = x_max - x_min
    #         y_local, x_local = np.ogrid[:local_h, :local_w]
            
    #         cy_local = center_y - y_min
    #         cx_local = center_x - x_min
            
    #         dist = np.sqrt((x_local - cx_local)**2 + (y_local - cy_local)**2)
    #         bubble_mask = dist <= radius
            
    #         if not np.any(bubble_mask):
    #             continue
            
    #         # ========== 1. 内部光学畸变（透镜效果）==========
    #         distortion_strength = 0.15 + coverage * 0.3
            
    #         # 预计算畸变映射
    #         map_y = np.zeros((local_h, local_w), dtype=np.float32)
    #         map_x = np.zeros((local_h, local_w), dtype=np.float32)
            
    #         for ly in range(local_h):
    #             for lx in range(local_w):
    #                 if bubble_mask[ly, lx]:
    #                     dx = lx - cx_local
    #                     dy = ly - cy_local
    #                     d = dist[ly, lx]
    #                     if d > 0:
    #                         factor = 1.0 - distortion_strength * (1 - (d / radius) ** 2)
    #                         map_x[ly, lx] = cx_local + dx * factor
    #                         map_y[ly, lx] = cy_local + dy * factor
    #                     else:
    #                         map_x[ly, lx] = lx
    #                         map_y[ly, lx] = ly
    #                 else:
    #                     map_x[ly, lx] = lx
    #                     map_y[ly, lx] = ly
            
    #         # 应用畸变
    #         if self._is_color_image(image):
    #             for c in range(image.shape[2]):
    #                 local_patch = artifact[y_min:y_max, x_min:x_max, c].copy()
    #                 warped = cv2.remap(local_patch, map_x, map_y, 
    #                                    interpolation=cv2.INTER_LINEAR,
    #                                    borderMode=cv2.BORDER_REFLECT)
    #                 # 只在气泡内部应用畸变
    #                 local_patch[bubble_mask] = warped[bubble_mask]
    #                 artifact[y_min:y_max, x_min:x_max, c] = local_patch
    #         else:
    #             local_patch = artifact[y_min:y_max, x_min:x_max].copy()
    #             warped = cv2.remap(local_patch, map_x, map_y,
    #                                interpolation=cv2.INTER_LINEAR,
    #                                borderMode=cv2.BORDER_REFLECT)
    #             local_patch[bubble_mask] = warped[bubble_mask]
    #             artifact[y_min:y_max, x_min:x_max] = local_patch
            
    #         # ========== 2. 边缘局部高光点（1-3个小亮点）==========
    #         num_highlights = rng.randint(1, 4)
    #         for _ in range(num_highlights):
    #             highlight_dist = radius * rng.uniform(0.6, 0.95)
    #             highlight_angle = rng.uniform(0, 2 * np.pi)
                
    #             hl_x = int(center_x + highlight_dist * np.cos(highlight_angle))
    #             hl_y = int(center_y + highlight_dist * np.sin(highlight_angle))
                
    #             hl_radius = max(2, int(radius * 0.2))
                
    #             # 创建高光区域
    #             for dy in range(-hl_radius, hl_radius + 1):
    #                 for dx in range(-hl_radius, hl_radius + 1):
    #                     py, px = hl_y + dy, hl_x + dx
    #                     if 0 <= py < h and 0 <= px < w:
    #                         d = np.sqrt(dx**2 + dy**2)
    #                         if d <= hl_radius:
    #                             intensity = np.exp(-d**2 / (2 * (hl_radius/2.5)**2))
    #                             intensity *= (0.3 + coverage * 0.6)
                                
    #                             if self._is_color_image(image):
    #                                 for c in range(image.shape[2]):
    #                                     max_val = image[:, :, c].max()
    #                                     artifact[py, px, c] = min(
    #                                         artifact[py, px, c] + intensity * max_val * 0.4,
    #                                         max_val
    #                                     )
    #                             else:
    #                                 max_val = image.max()
    #                                 artifact[py, px] = min(
    #                                     artifact[py, px] + intensity * max_val * 0.4,
    #                                     max_val
    #                                 )
            
    #         # ========== 3. 边缘轻微增亮（凸起感）==========
    #         edge_width = max(2, radius // 4)
    #         edge_mask = (dist > radius - edge_width) & (dist <= radius)
            
    #         if np.any(edge_mask):
    #             edge_intensity = 0.08 + coverage * 0.12
    #             if self._is_color_image(image):
    #                 for c in range(image.shape[2]):
    #                     local_region = artifact[y_min:y_max, x_min:x_max, c]
    #                     local_region[edge_mask] *= (1 + edge_intensity)
    #                     artifact[y_min:y_max, x_min:x_max, c] = np.clip(
    #                         local_region, 0, image[:, :, c].max()
    #                     )
    #             else:
    #                 local_region = artifact[y_min:y_max, x_min:x_max]
    #                 local_region[edge_mask] *= (1 + edge_intensity)
    #                 artifact[y_min:y_max, x_min:x_max] = np.clip(
    #                     local_region, 0, image.max()
    #                 )
        
    #     return artifact.astype(np.float32)
    
    # enhence bubble
    def _endo_bubbles(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        内窥镜气泡（向量化优化版）
        """
        h, w = image.shape[:2]
        artifact = image.copy().astype(np.float32)
        
        coverage = self.severity_params['endo_bubble_coverage'][severity]
        rng = np.random.RandomState(self.seed + severity * 600)
        
        num_bubbles = max(3, int(coverage * 80))
        
        for _ in range(num_bubbles):
            center_y = rng.randint(h // 10, 9 * h // 10)
            center_x = rng.randint(w // 10, 9 * w // 10)
            radius = rng.randint(max(8, int(min(h, w) * 0.03)), 
                                 max(20, int(min(h, w) * 0.12)))
            
            # 局部区域
            y_min = max(0, center_y - radius - 2)
            y_max = min(h, center_y + radius + 3)
            x_min = max(0, center_x - radius - 2)
            x_max = min(w, center_x + radius + 3)
            
            if y_max <= y_min or x_max <= x_min:
                continue
            
            local_h = y_max - y_min
            local_w = x_max - x_min
            cy_local = center_y - y_min
            cx_local = center_x - x_min
            
            # 向量化计算距离
            y_coords, x_coords = np.mgrid[:local_h, :local_w]
            dx = x_coords - cx_local
            dy = y_coords - cy_local
            dist = np.sqrt(dx**2 + dy**2)
            bubble_mask = dist <= radius
            
            if not np.any(bubble_mask):
                continue
            
            # ========== 1. 向量化畸变映射 ==========
            distortion_strength = 0.25 + coverage * 0.6
            
            # 避免除零
            dist_safe = np.where(dist > 0, dist, 1)
            factor = np.where(dist > 0, 
                              1.0 - distortion_strength * (1 - (dist / radius) ** 2),
                              1.0)
            
            map_x = (cx_local + dx * factor).astype(np.float32)
            map_y = (cy_local + dy * factor).astype(np.float32)
            
            # 非气泡区域保持原坐标
            map_x = np.where(bubble_mask, map_x, x_coords.astype(np.float32))
            map_y = np.where(bubble_mask, map_y, y_coords.astype(np.float32))
            
            # 应用畸变
            if self._is_color_image(image):
                for c in range(image.shape[2]):
                    local_patch = artifact[y_min:y_max, x_min:x_max, c].copy()
                    warped = cv2.remap(local_patch, map_x, map_y, 
                                       interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT)
                    local_patch[bubble_mask] = warped[bubble_mask]
                    artifact[y_min:y_max, x_min:x_max, c] = local_patch
            else:
                local_patch = artifact[y_min:y_max, x_min:x_max].copy()
                warped = cv2.remap(local_patch, map_x, map_y,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
                local_patch[bubble_mask] = warped[bubble_mask]
                artifact[y_min:y_max, x_min:x_max] = local_patch
            
            # ========== 2. 向量化高光点 ==========
            num_highlights = rng.randint(2, 5)
            for _ in range(num_highlights):
                highlight_dist = radius * rng.uniform(0.5, 0.95)
                highlight_angle = rng.uniform(0, 2 * np.pi)
                
                hl_x = center_x + highlight_dist * np.cos(highlight_angle)
                hl_y = center_y + highlight_dist * np.sin(highlight_angle)
                hl_radius = max(3, int(radius * 0.25))
                
                # 高光区域边界
                hl_y_min = max(0, int(hl_y - hl_radius))
                hl_y_max = min(h, int(hl_y + hl_radius + 1))
                hl_x_min = max(0, int(hl_x - hl_radius))
                hl_x_max = min(w, int(hl_x + hl_radius + 1))
                
                if hl_y_max <= hl_y_min or hl_x_max <= hl_x_min:
                    continue
                
                # 向量化计算高光
                hy, hx = np.mgrid[hl_y_min:hl_y_max, hl_x_min:hl_x_max]
                hl_dist = np.sqrt((hx - hl_x)**2 + (hy - hl_y)**2)
                hl_mask = hl_dist <= hl_radius
                
                intensity = np.exp(-hl_dist**2 / (2 * (hl_radius/2.5)**2))
                intensity *= (0.6 + coverage * 1.2)
                
                if self._is_color_image(image):
                    for c in range(image.shape[2]):
                        max_val = image[:, :, c].max()
                        addition = intensity * max_val * 0.7 * hl_mask
                        artifact[hl_y_min:hl_y_max, hl_x_min:hl_x_max, c] = np.minimum(
                            artifact[hl_y_min:hl_y_max, hl_x_min:hl_x_max, c] + addition,
                            max_val
                        )
                else:
                    max_val = image.max()
                    addition = intensity * max_val * 0.7 * hl_mask
                    artifact[hl_y_min:hl_y_max, hl_x_min:hl_x_max] = np.minimum(
                        artifact[hl_y_min:hl_y_max, hl_x_min:hl_x_max] + addition,
                        max_val
                    )
            
            # ========== 3. 向量化边缘增亮 ==========
            edge_width = max(3, radius // 3)
            edge_mask = (dist > radius - edge_width) & (dist <= radius)
            
            if np.any(edge_mask):
                edge_intensity = 0.15 + coverage * 0.25
                if self._is_color_image(image):
                    for c in range(image.shape[2]):
                        local_region = artifact[y_min:y_max, x_min:x_max, c]
                        local_region[edge_mask] *= (1 + edge_intensity)
                        artifact[y_min:y_max, x_min:x_max, c] = np.clip(
                            local_region, 0, image[:, :, c].max()
                        )
                else:
                    local_region = artifact[y_min:y_max, x_min:x_max]
                    local_region[edge_mask] *= (1 + edge_intensity)
                    artifact[y_min:y_max, x_min:x_max] = np.clip(
                        local_region, 0, image.max()
                    )
            
            # ========== 4. 向量化内部提亮 ==========
            inner_mask = dist <= radius * 0.7
            if np.any(inner_mask):
                brightness_shift = 0.05 + coverage * 0.1
                if self._is_color_image(image):
                    for c in range(image.shape[2]):
                        local_region = artifact[y_min:y_max, x_min:x_max, c]
                        channel_range = image[:, :, c].max() - image[:, :, c].min()
                        local_region[inner_mask] += brightness_shift * channel_range
                        artifact[y_min:y_max, x_min:x_max, c] = np.clip(
                            local_region, 0, image[:, :, c].max()
                        )
                else:
                    local_region = artifact[y_min:y_max, x_min:x_max]
                    img_range = image.max() - image.min()
                    local_region[inner_mask] += brightness_shift * img_range
                    artifact[y_min:y_max, x_min:x_max] = np.clip(
                        local_region, 0, image.max()
                    )
        
        return artifact.astype(np.float32)
    
    def _endo_blood(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        内窥镜血液遮挡 - 出血或残留血液
        
        物理特性：
        - 血液呈连续区域分布，不是离散像素
        - 颜色偏红（RGB中R通道保持，G/B通道降低）
        - 边缘有渐变过渡
        """
        h, w = image.shape[:2]
        artifact = image.copy()
        
        # 从severity_params获取覆盖率
        coverage = self.severity_params['endo_blood_coverage'][severity]
        
        rng = np.random.RandomState(self.seed + severity * 700)
        
        # 创建血液区域mask（使用多个椭圆形斑块）
        blood_mask = np.zeros((h, w), dtype=np.float32)
        
        num_patches = max(1, int(coverage * 20))
        
        for _ in range(num_patches):
            center_y = rng.randint(0, h)
            center_x = rng.randint(0, w)
            
            # 椭圆形状的血液斑块
            radius_y = rng.randint(max(5, h // 20), max(10, h // 8))
            radius_x = rng.randint(max(5, w // 20), max(10, w // 8))
            
            y, x = np.ogrid[:h, :w]
            ellipse_dist = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2
            
            # 软边缘
            patch = np.exp(-ellipse_dist * 2)
            patch = np.clip(patch, 0, 1)
            
            # 随机强度
            patch_intensity = rng.uniform(0.5, 1.0)
            blood_mask = np.maximum(blood_mask, patch * patch_intensity)
        
        # 限制总覆盖面积
        blood_mask = np.clip(blood_mask * coverage * 5, 0, 1)
        
        # 应用血液效果
        if self._is_color_image(image):
            # 彩色图像：降低G和B通道，保持R通道
            artifact[:, :, 0] = artifact[:, :, 0] * (1 - blood_mask * 0.1)  # R通道轻微降低
            artifact[:, :, 1] = artifact[:, :, 1] * (1 - blood_mask * 0.5)  # G通道明显降低
            artifact[:, :, 2] = artifact[:, :, 2] * (1 - blood_mask * 0.5)  # B通道明显降低
        else:
            # 灰度图像：整体变暗
            artifact = artifact * (1 - blood_mask * 0.4)
        
        return artifact.astype(np.float32)
    
    def _endo_saturation(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        内窥镜过度曝光 - 光源过近导致的局部过曝
        
        物理特性：
        - 过曝区域是连续的，通常靠近图像中心或光源位置
        - 呈圆形或椭圆形渐变
        - 中心完全饱和，边缘渐变过渡
        """
        h, w = image.shape[:2]
        artifact = image.copy()
        
        # 从severity_params获取过曝比例
        saturation_ratio = self.severity_params['endo_saturation_ratio'][severity]
        
        rng = np.random.RandomState(self.seed + severity * 800)
        
        # 创建过曝区域
        saturation_mask = np.zeros((h, w), dtype=np.float32)
        
        # 过曝斑块数量
        num_spots = max(1, int(saturation_ratio * 15))
        
        for _ in range(num_spots):
            # 过曝位置（偏向中心区域，模拟光源位置）
            center_y = rng.randint(h // 4, 3 * h // 4)
            center_x = rng.randint(w // 4, 3 * w // 4)
            
            # 过曝区域大小
            radius = rng.randint(max(10, min(h, w) // 15), max(20, min(h, w) // 6))
            
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # 软边缘的过曝区域
            spot = np.exp(-(dist / radius) ** 2 * 2)
            spot = np.clip(spot, 0, 1)
            
            saturation_mask = np.maximum(saturation_mask, spot)
        
        # 限制总过曝面积
        saturation_mask = np.clip(saturation_mask * saturation_ratio * 8, 0, 1)
        
        # 应用过曝效果（向最大值靠近）
        if self._is_color_image(image):
            for c in range(image.shape[2]):
                max_val = image[:, :, c].max()
                artifact[:, :, c] = artifact[:, :, c] * (1 - saturation_mask) + max_val * saturation_mask
        else:
            max_val = image.max()
            artifact = artifact * (1 - saturation_mask) + max_val * saturation_mask
        
        return artifact.astype(np.float32)

    
    # === OCT Perturbations ===
    # === OCT Perturbations ===
    
    def _oct_shadow(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        OCT阴影伪影 - 玻璃体混浊、出血等遮挡造成的垂直信号丢失
        
        物理特性：
        - 垂直黑色条纹（信号完全丢失）
        - 可能有多条
        - 边缘可能略有过渡
        """
        h, w = image.shape[:2]
        artifact = image.copy().astype(np.float32)
        
        shadow_width_ratio = self.severity_params['oct_shadow_width'][severity]
        rng = np.random.RandomState(self.seed + severity * 900)
        
        # 阴影数量（1-3条）
        num_shadows = rng.randint(1, min(4, severity + 1))
        
        for _ in range(num_shadows):
            # 阴影宽度
            shadow_width = int(w * shadow_width_ratio * rng.uniform(0.7, 1.3))
            shadow_width = max(3, shadow_width)
            
            # 阴影位置（避开边缘）
            center_x = rng.randint(shadow_width, w - shadow_width)
            
            left = max(0, center_x - shadow_width // 2)
            right = min(w, center_x + shadow_width // 2)
            
            # 边缘过渡宽度
            edge_width = max(2, shadow_width // 6)
            
            # 创建阴影mask
            x_coords = np.arange(w)
            shadow_mask = np.ones(w, dtype=np.float32)
            
            # 中心区域完全黑
            shadow_mask[left + edge_width:right - edge_width] = 0.0
            
            # 左边缘渐变
            left_edge = (x_coords >= left) & (x_coords < left + edge_width)
            if np.any(left_edge):
                shadow_mask[left_edge] = (x_coords[left_edge] - left) / edge_width
                shadow_mask[left_edge] = 1 - shadow_mask[left_edge]
            
            # 右边缘渐变
            right_edge = (x_coords >= right - edge_width) & (x_coords < right)
            if np.any(right_edge):
                shadow_mask[right_edge] = (x_coords[right_edge] - (right - edge_width)) / edge_width
            
            # 应用阴影（整个垂直方向）
            shadow_mask_2d = np.tile(shadow_mask, (h, 1))
            
            if self._is_color_image(image):
                for c in range(image.shape[2]):
                    artifact[:, :, c] *= shadow_mask_2d
            else:
                artifact *= shadow_mask_2d
        
        return artifact.astype(np.float32)
    
    def _oct_blink(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        OCT眨眼伪影 - 扫描时眨眼导致的水平黑色条纹
        
        物理特性：
        - 水平黑色条纹带
        - 信号完全丢失
        - 通常有一定厚度
        """
        h, w = image.shape[:2]
        artifact = image.copy().astype(np.float32)
        
        blink_count = int(self.severity_params['oct_blink_count'][severity])
        rng = np.random.RandomState(self.seed + severity * 910)
        
        for _ in range(blink_count):
            # 条纹厚度（占图像高度的比例）
            band_height = rng.randint(max(2, h // 50), max(5, h // 20))
            
            # 条纹位置
            band_y = rng.randint(band_height, h - band_height)
            
            y_start = max(0, band_y - band_height // 2)
            y_end = min(h, band_y + band_height // 2)
            
            # 边缘过渡
            edge_h = max(1, band_height // 4)
            
            for y in range(y_start, y_end):
                # 计算该行的衰减因子
                dist_from_center = abs(y - band_y)
                if dist_from_center < band_height // 2 - edge_h:
                    factor = 0.0  # 完全黑
                else:
                    # 边缘渐变
                    factor = (dist_from_center - (band_height // 2 - edge_h)) / edge_h
                    factor = min(1.0, factor)
                
                if self._is_color_image(image):
                    for c in range(image.shape[2]):
                        artifact[y, :, c] *= factor
                else:
                    artifact[y, :] *= factor
        
        return artifact.astype(np.float32)
    
    def _oct_motion(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        OCT运动伪影 - 眼球运动导致的图像水平错位
        
        物理特性：
        - 图像某部分水平移位
        - 在移位边界处有明显断裂
        - 类似"错位"效果
        """
        h, w = image.shape[:2]
        artifact = image.copy().astype(np.float32)
        
        shift_ratio = self.severity_params['oct_motion_shift'][severity]
        rng = np.random.RandomState(self.seed + severity * 920)
        
        # 错位次数
        num_shifts = rng.randint(1, severity + 1)
        
        for _ in range(num_shifts):
            # 错位区域的起始和结束行
            region_height = rng.randint(h // 8, h // 3)
            region_start = rng.randint(0, h - region_height)
            region_end = region_start + region_height
            
            # 错位量（像素）
            max_shift = int(w * shift_ratio)
            shift_amount = rng.randint(-max_shift, max_shift + 1)
            
            if shift_amount == 0:
                continue
            
            # 应用水平错位
            if self._is_color_image(image):
                for c in range(image.shape[2]):
                    region = artifact[region_start:region_end, :, c].copy()
                    artifact[region_start:region_end, :, c] = np.roll(region, shift_amount, axis=1)
                    
                    # 填充边缘（错位后露出的部分用边缘值填充）
                    if shift_amount > 0:
                        artifact[region_start:region_end, :shift_amount, c] = artifact[region_start:region_end, shift_amount:shift_amount+1, c]
                    else:
                        artifact[region_start:region_end, shift_amount:, c] = artifact[region_start:region_end, shift_amount-1:shift_amount, c]
            else:
                region = artifact[region_start:region_end, :].copy()
                artifact[region_start:region_end, :] = np.roll(region, shift_amount, axis=1)
                
                if shift_amount > 0:
                    artifact[region_start:region_end, :shift_amount] = artifact[region_start:region_end, shift_amount:shift_amount+1]
                else:
                    artifact[region_start:region_end, shift_amount:] = artifact[region_start:region_end, shift_amount-1:shift_amount]
        
        return artifact.astype(np.float32)
    
    def _oct_defocus(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        OCT离焦伪影 - 焦点不准导致的图像模糊
        """
        sigma = self.severity_params['gaussian_blur'][severity]
        
        if self._is_color_image(image):
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = gaussian_filter(image[:, :, c], sigma=sigma)
            return result
        else:
            return gaussian_filter(image, sigma=sigma)
    
    def _oct_speckle(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        OCT散斑噪声 - OCT成像固有的散斑干涉噪声
        
        物理特性：
        - 乘性噪声（与信号强度相关）
        - 颗粒状外观
        - OCT特有的干涉斑点
        """
        h, w = image.shape[:2]
        
        speckle_intensity = self.severity_params['oct_speckle_intensity'][severity]
        rng = np.random.RandomState(self.seed + severity * 930)
        
        # 生成散斑噪声（乘性）
        # OCT散斑通常服从瑞利分布，这里用对数正态近似
        speckle = rng.lognormal(mean=0, sigma=speckle_intensity, size=(h, w))
        
        # 归一化到合理范围
        speckle = speckle / np.mean(speckle)
        
        if self._is_color_image(image):
            artifact = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                artifact[:, :, c] = image[:, :, c] * speckle
                artifact[:, :, c] = np.clip(artifact[:, :, c], image[:, :, c].min(), image[:, :, c].max())
        else:
            artifact = image * speckle
            artifact = np.clip(artifact, image.min(), image.max())
        
        return artifact.astype(np.float32)
    
    # === Pathology Perturbations ===
    def _path_stain_variation(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        病理切片H&E染色变化
        
        原理：在HSV颜色空间调整色调/饱和度/亮度
        
        关键：intensity 从 self.severity_params['path_stain_intensity'] 读取
        这样 adaptive 模式的二分搜索才能生效
        """
        h, w = image.shape[:2]
        rng = np.random.RandomState(self.seed + severity * 1000)
        
        # ====== 关键：从 severity_params 获取强度参数 ======
        intensity = self.severity_params['path_stain_intensity'][severity]
        
        # 灰度图像处理
        if not self._is_color_image(image):
            brightness_factor = rng.uniform(1 - intensity, 1 + intensity)
            artifact = image * brightness_factor
            return np.clip(artifact, image.min(), image.max()).astype(np.float32)
        
        # === 彩色图像：HSV空间调整 ===
        
        # 归一化到0-255 uint8
        img_min, img_max = image.min(), image.max()
        if img_max - img_min > 1e-8:
            img_norm = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            return image.copy()
        
        # RGB -> HSV
        hsv = cv2.cvtColor(img_norm, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 1. Hue (色调) 偏移 - intensity 越大偏移越大
        hue_shift = rng.uniform(-intensity * 40, intensity * 40)  # 范围扩大
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # 2. Saturation (饱和度) 变化
        sat_factor = rng.uniform(1 - intensity, 1 + intensity)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        # 3. Value (亮度) 变化
        val_factor = rng.uniform(1 - intensity * 0.6, 1 + intensity * 0.6)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 255)
        
        # 4. 空间不均匀性（当 intensity > 0.15 时）
        if intensity > 0.15:
            field_size = max(2, min(h, w) // 30)
            spatial_field = rng.rand(field_size, field_size).astype(np.float32)
            spatial_field = cv2.resize(spatial_field, (w, h), interpolation=cv2.INTER_CUBIC)
            spatial_field = gaussian_filter(spatial_field, sigma=min(h, w) / 8)
            spatial_field = spatial_field - spatial_field.mean()
            spatial_field = 1.0 + spatial_field * intensity * 0.8
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * spatial_field, 0, 255)
        
        # HSV -> RGB
        hsv_uint8 = np.clip(hsv, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB).astype(np.float32)
        
        # 恢复原始范围
        result = result / 255.0 * (img_max - img_min) + img_min
        
        return result.astype(np.float32)


    def _path_bubble_artifacts(self, image: np.ndarray, severity: int) -> np.ndarray:
        """
        病理切片气泡伪影 - 基于文献的真实版
        
        物理特性（来自病理学文献）:
        1. 盖玻片下气泡: "tiny spherical features", 透明/淡色区域
        2. 气泡遮挡下方组织，呈现为接近背景色的圆形区域
        3. 边缘有折射性亮环或暗环（highly refractile lines）
        4. 塌陷气泡（Collapsed bubble artifact）: 留下"cracked areas"裂纹
        
        参数来源: self.severity_params['path_bubble_count'] 控制气泡数量
                  self.severity_params['path_bubble_cracks'] 控制裂纹数量
        """
        h, w = image.shape[:2]
        artifact = image.copy().astype(np.float32)
        
        rng = np.random.RandomState(self.seed + severity * 1100)
        
        # 从参数获取气泡数量（支持自适应模式）
        num_bubbles = int(self.severity_params['path_bubble_count'][severity])
        
        # 裂纹数量（塌陷气泡），如果没有定义则使用默认值
        if 'path_bubble_cracks' in self.severity_params:
            num_cracks = int(self.severity_params['path_bubble_cracks'][severity])
        else:
            num_cracks = max(0, severity - 1)  # severity 1无裂纹，5有4条
        
        # 计算图像的背景色（病理切片通常背景偏白/粉色）
        if self._is_color_image(image):
            bg_color = np.percentile(image, 92, axis=(0, 1))
        else:
            bg_color = np.percentile(image, 92)
        
        # === 1. 完整气泡（盖玻片下的气泡）===
        base_size = min(h, w)
        
        for _ in range(num_bubbles):
            # 气泡位置（避开边缘）
            margin = max(15, base_size // 12)
            if h <= 2 * margin or w <= 2 * margin:
                continue
            
            center_y = rng.randint(margin, h - margin)
            center_x = rng.randint(margin, w - margin)
            
            # 气泡半径（基于气泡数量参数动态调整）
            size_factor = num_bubbles / 5.0
            min_radius = max(6, int(base_size * (0.015 + size_factor * 0.01)))
            max_radius = max(min_radius + 8, int(base_size * (0.04 + size_factor * 0.03)))
            radius = rng.randint(min_radius, max_radius + 1)
            
            # 局部区域
            pad = 5
            y_min = max(0, center_y - radius - pad)
            y_max = min(h, center_y + radius + pad)
            x_min = max(0, center_x - radius - pad)
            x_max = min(w, center_x + radius + pad)
            
            local_h = y_max - y_min
            local_w = x_max - x_min
            
            if local_h <= 0 or local_w <= 0:
                continue
                
            cy_local = center_y - y_min
            cx_local = center_x - x_min
            
            # 创建距离场
            y_coords, x_coords = np.ogrid[:local_h, :local_w]
            dist = np.sqrt((x_coords - cx_local)**2 + (y_coords - cy_local)**2)
            
            # 气泡掩码
            bubble_mask = dist <= radius
            
            if not np.any(bubble_mask):
                continue
            
            # 边缘区域定义
            edge_width = max(2, radius // 5)
            edge_outer = (dist > radius - edge_width) & (dist <= radius)
            edge_inner = (dist > radius - edge_width * 2) & (dist <= radius - edge_width)
            inner_core = dist <= radius - edge_width * 2
            
            # === 气泡内部填充效果 ===
            # 中心到边缘的渐变因子
            fill_gradient = 1.0 - (dist / radius) ** 1.5
            fill_gradient = np.clip(fill_gradient, 0, 1)
            
            # 填充强度（中心更强）
            fill_strength = fill_gradient * 0.45
            
            if self._is_color_image(image):
                for c in range(image.shape[2]):
                    local_patch = artifact[y_min:y_max, x_min:x_max, c].copy()
                    original_patch = local_patch.copy()
                    
                    # 计算填充色（接近背景色但略透明）
                    fill_color = bg_color[c]
                    
                    # 气泡核心：较强的背景色填充
                    if np.any(inner_core):
                        blend = fill_strength[inner_core]
                        local_patch[inner_core] = (
                            original_patch[inner_core] * (1 - blend) + 
                            fill_color * blend
                        )
                    
                    # 内边缘：轻微亮环（折射效果）
                    if np.any(edge_inner):
                        local_patch[edge_inner] = np.minimum(
                            local_patch[edge_inner] * 1.08,
                            image[:, :, c].max()
                        )
                    
                    # 外边缘：暗环（阴影/折射）
                    if np.any(edge_outer):
                        edge_factor = 0.82 + 0.1 * (dist[edge_outer] - (radius - edge_width)) / edge_width
                        local_patch[edge_outer] *= edge_factor
                    
                    artifact[y_min:y_max, x_min:x_max, c] = local_patch
            else:
                local_patch = artifact[y_min:y_max, x_min:x_max].copy()
                original_patch = local_patch.copy()
                
                if np.any(inner_core):
                    blend = fill_strength[inner_core]
                    local_patch[inner_core] = (
                        original_patch[inner_core] * (1 - blend) + 
                        bg_color * blend
                    )
                
                if np.any(edge_inner):
                    local_patch[edge_inner] = np.minimum(
                        local_patch[edge_inner] * 1.08,
                        image.max()
                    )
                
                if np.any(edge_outer):
                    edge_factor = 0.82 + 0.1 * (dist[edge_outer] - (radius - edge_width)) / edge_width
                    local_patch[edge_outer] *= edge_factor
                
                artifact[y_min:y_max, x_min:x_max] = local_patch
        
        # === 2. 塌陷气泡裂纹（Collapsed bubble artifact）===
        for _ in range(num_cracks):
            # 塌陷位置
            center_y = rng.randint(h // 10, 9 * h // 10)
            center_x = rng.randint(w // 10, 9 * w // 10)
            
            # 裂纹辐射范围
            crack_radius = rng.randint(
                max(12, base_size // 25), 
                max(30, base_size // 10)
            )
            
            # 从中心辐射的裂纹条数
            num_lines = rng.randint(3, 6)
            
            for i in range(num_lines):
                # 裂纹角度（带随机偏移）
                base_angle = 2 * np.pi * i / num_lines
                angle = base_angle + rng.uniform(-0.4, 0.4)
                
                # 裂纹长度
                line_length = int(crack_radius * rng.uniform(0.4, 1.1))
                
                # 裂纹宽度
                line_width = rng.randint(1, 2)
                
                # 沿裂纹绘制
                for t in range(line_length):
                    # 添加轻微弯曲
                    wobble_x = rng.uniform(-1.5, 1.5)
                    wobble_y = rng.uniform(-1.5, 1.5)
                    
                    px = int(center_x + t * np.cos(angle) + wobble_x)
                    py = int(center_y + t * np.sin(angle) + wobble_y)
                    
                    # 绘制裂纹像素
                    for dx in range(-line_width, line_width + 1):
                        for dy in range(-line_width, line_width + 1):
                            if dx * dx + dy * dy <= line_width * line_width:
                                nx, ny = px + dx, py + dy
                                if 0 <= ny < h and 0 <= nx < w:
                                    # 裂纹处：染色改变（文献描述）
                                    # 随机变亮或变暗
                                    if rng.random() > 0.4:
                                        factor = rng.uniform(0.65, 0.85)  # 变暗
                                    else:
                                        factor = rng.uniform(1.1, 1.25)   # 变亮
                                    
                                    if self._is_color_image(image):
                                        for c in range(image.shape[2]):
                                            artifact[ny, nx, c] = np.clip(
                                                artifact[ny, nx, c] * factor,
                                                0, image[:, :, c].max()
                                            )
                                    else:
                                        artifact[ny, nx] = np.clip(
                                            artifact[ny, nx] * factor,
                                            0, image.max()
                                        )
            
            # 塌陷区域中心染色改变（"altered staining"）
            collapse_radius = crack_radius // 3
            y_min = max(0, center_y - collapse_radius)
            y_max = min(h, center_y + collapse_radius)
            x_min = max(0, center_x - collapse_radius)
            x_max = min(w, center_x + collapse_radius)
            
            if y_max > y_min and x_max > x_min:
                local_h = y_max - y_min
                local_w = x_max - x_min
                cy_local = center_y - y_min
                cx_local = center_x - x_min
                
                y_coords, x_coords = np.ogrid[:local_h, :local_w]
                dist = np.sqrt((x_coords - cx_local)**2 + (y_coords - cy_local)**2)
                
                collapse_mask = dist < collapse_radius
                if np.any(collapse_mask):
                    # 中心区域轻微变淡（组织与玻片粘附不良）
                    fade = 1.0 - 0.12 * (1 - dist[collapse_mask] / collapse_radius)
                    
                    if self._is_color_image(image):
                        for c in range(image.shape[2]):
                            local_patch = artifact[y_min:y_max, x_min:x_max, c]
                            local_patch[collapse_mask] *= fade
                    else:
                        local_patch = artifact[y_min:y_max, x_min:x_max]
                        local_patch[collapse_mask] *= fade
        
        return artifact.astype(np.float32)

    
    # === CT Perturbations ===
    def _ct_metal_artifacts(self, image: np.ndarray, severity: int) -> np.ndarray:
        h, w = image.shape[:2]
        artifact = image.copy()
        metal_y = np.random.randint(h // 4, 3 * h // 4)
        metal_x = np.random.randint(w // 4, 3 * w // 4)
        num_streaks = 8 + severity * 4
        max_length = min(h, w) // 2
        intensity = severity * 0.15
        for i in range(num_streaks):
            angle = (2 * np.pi * i) / num_streaks
            dx, dy = np.cos(angle), np.sin(angle)
            for t in range(max_length):
                x = int(metal_x + t * dx)
                y = int(metal_y + t * dy)
                if 0 <= y < h and 0 <= x < w:
                    streak_intensity = intensity * (1 - t / max_length)
                    factor = (1 + streak_intensity) if i % 2 == 0 else (1 - streak_intensity)
                    if self._is_color_image(image):
                        for c in range(image.shape[2]):
                            artifact[y, x, c] *= factor
                    else:
                        artifact[y, x] *= factor
        if self._is_color_image(image):
            for c in range(image.shape[2]):
                artifact[:, :, c] = np.clip(artifact[:, :, c], image[:, :, c].min(), image[:, :, c].max())
        else:
            artifact = np.clip(artifact, image.min(), image.max())
        return artifact
    
    def _ct_beam_hardening(self, image: np.ndarray, severity: int) -> np.ndarray:
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        distance = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        cupping_strength = severity * 0.1
        cupping_map = 1.0 - cupping_strength * (distance / max_dist)**2
        if self._is_color_image(image):
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = image[:, :, c] * cupping_map
            return result
        else:
            return image * cupping_map
    
    def _ct_photon_starvation(self, image: np.ndarray, severity: int) -> np.ndarray:
        noise_std_map = {1: 0.03, 2: 0.06, 3: 0.09, 4: 0.12, 5: 0.15}
        noise_std = noise_std_map[severity]
        if self._is_color_image(image):
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                noise = np.random.normal(0, noise_std * np.sqrt(np.abs(channel) + 1), channel.shape)
                result[:, :, c] = np.clip(channel + noise, channel.min(), channel.max())
            return result
        else:
            noise = np.random.normal(0, noise_std * np.sqrt(np.abs(image) + 1), image.shape)
            return np.clip(image + noise, image.min(), image.max())
    
    # ====================== CSV & Dataset Generation ======================
    def _init_csv_file(self, csv_path: str):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'perturbation_type', 'severity', 'ssim', 'mse'])
    
    def _append_to_csv(self, csv_path: str, file_name: str, pert_type: str, severity: int, ssim_val: float, mse_val: float):
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([file_name, pert_type, severity, f'{ssim_val:.6f}', f'{mse_val:.8f}'])
    
    def _init_csv_file_adaptive(self, csv_path: str):
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'perturbation_type', 'severity', 'ssim', 'mse', 'param_used'])
    
    def _append_to_csv_adaptive(self, csv_path: str, file_name: str, pert_type: str, severity: int, ssim_val: float, mse_val: float, param_used: float):
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            param_str = f'{param_used:.6f}' if param_used is not None else 'N/A'
            writer.writerow([file_name, pert_type, severity, f'{ssim_val:.6f}', f'{mse_val:.8f}', param_str])
    
    def generate_perturbation_dataset(self, dataset_config: Dict[str, Any], 
                                     perturbation_config: Dict[str, List[str]],
                                     output_root: str = None,
                                     adaptive_mode: bool = False,
                                     ssim_targets: Dict[int, Tuple[float, float]] = None,
                                     calibration_samples: int = 1):
        """
        生成扰动数据集
        calibration_samples: 用于校准参数的图像数量（默认1）
        """
        dataset_name = dataset_config['name'].lower()
        modality = self.detect_modality_from_path(dataset_name)
        
        if output_root is None:
            img_dir = Path(dataset_config['resized_img_dir'])
            # 修改默认输出路径，与dataset_config.json中的root_path保持一致
            output_dir = img_dir.parent / "perturbed_datasets" / "full_image_perturbations"
        else:
            output_dir = Path(output_root)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir.parent / f"{dataset_name.replace(' ', '_')}_metrics.csv"
        
        if adaptive_mode:
            self._init_csv_file_adaptive(str(csv_path))
        else:
            self._init_csv_file(str(csv_path))
        
        if adaptive_mode:
            if ssim_targets is None:
                ssim_targets = self.ssim_targets if self.ssim_targets else self.default_ssim_targets
            print(f"Adaptive mode enabled with SSIM targets: {ssim_targets}")
            print(f"Using first {calibration_samples} image(s) for parameter calibration")
            self.clear_param_cache()
        
        print(f"Creating perturbations for {dataset_name} ({modality}) in: {output_dir}")
        print(f"Metrics will be saved to: {csv_path}")
        
        image_extensions = dataset_config.get('image_extensions', ['png', 'jpg'])
        image_files = []
        for ext in image_extensions:
            if ext.startswith('.'):
                ext = ext[1:]
            pattern = os.path.join(dataset_config['resized_img_dir'], f"*.{ext}")
            found_files = glob.glob(pattern)
            image_files.extend(found_files)
            print(f"  Found {len(found_files)} .{ext} files")
        
        if not image_files:
            raise ValueError(f"No images found in directory: {dataset_config['resized_img_dir']}")
        
        print(f"Found {len(image_files)} images to process")
        
        all_perturbations = perturbation_config.get('base', []) + perturbation_config.get('modality_specific', [])
        for pert_type in all_perturbations:
            for severity in [0, 1,3,5]:
                severity_dir = output_dir / pert_type / str(severity)
                severity_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_file in enumerate(image_files):
            try:
                is_npy = dataset_config.get('image_extensions', [''])[0].lower() == 'npy'
                npy_format = dataset_config.get('npy_format', 'single')
                image = self.load_image(img_file, is_npy=is_npy, npy_format=npy_format)
                filename = os.path.basename(img_file)
                
                for pert_type in perturbation_config.get('base', []):
                    for severity in [0,1,3,5]:
                        try:
                            if adaptive_mode:
                                corrupted, ssim_val, param_used = self.apply_perturbation_adaptive(
                                    image, modality, pert_type, severity, ssim_targets, use_cache=True)
                                mse_val = self.compute_mse(image, corrupted)
                            else:
                                corrupted = self.apply_perturbation(image, modality, pert_type, severity)
                                ssim_val = self.compute_ssim(image, corrupted)
                                mse_val = self.compute_mse(image, corrupted)
                                param_used = None
                            
                            output_file = output_dir / pert_type / str(severity) / filename
                            if not str(output_file).lower().endswith('.png'):
                                output_file = output_file.with_suffix('.png')
                            self.save_image(corrupted, str(output_file))
                            
                            output_filename = str(output_file.relative_to(output_dir))
                            if adaptive_mode:
                                self._append_to_csv_adaptive(str(csv_path), output_filename, pert_type, severity, ssim_val, mse_val, param_used)
                            else:
                                self._append_to_csv(str(csv_path), output_filename, pert_type, severity, ssim_val, mse_val)
                        except Exception as e:
                            warnings.warn(f"Error applying {pert_type} s{severity} to {filename}: {e}")
                
                for pert_type in perturbation_config.get('modality_specific', []):
                    for severity in [0,1,3,5]:
                        try:
                            if adaptive_mode:
                                corrupted, ssim_val, param_used = self.apply_perturbation_adaptive(
                                    image, modality, pert_type, severity, ssim_targets, use_cache=True)
                                mse_val = self.compute_mse(image, corrupted)
                            else:
                                corrupted = self.apply_perturbation(image, modality, pert_type, severity)
                                ssim_val = self.compute_ssim(image, corrupted)
                                mse_val = self.compute_mse(image, corrupted)
                                param_used = None
                            
                            output_file = output_dir / pert_type / str(severity) / filename
                            if not str(output_file).lower().endswith('.png'):
                                output_file = output_file.with_suffix('.png')
                            self.save_image(corrupted, str(output_file))
                            
                            output_filename = str(output_file.relative_to(output_dir))
                            if adaptive_mode:
                                self._append_to_csv_adaptive(str(csv_path), output_filename, pert_type, severity, ssim_val, mse_val, param_used)
                            else:
                                self._append_to_csv(str(csv_path), output_filename, pert_type, severity, ssim_val, mse_val)
                        except Exception as e:
                            warnings.warn(f"Error applying {pert_type} s{severity} to {filename}: {e}")
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(image_files)} images")
                
                if adaptive_mode and i == 0:
                    print(f"\n📌 Calibrated parameters (from first image):")
                    for pt, severities in self.cached_params.items():
                        params_str = ", ".join([f"s{s}={p:.4f}" for s, p in sorted(severities.items())])
                        print(f"   {pt}: {params_str}")
                    print()
            except Exception as e:
                warnings.warn(f"Error processing {img_file}: {e}")
                continue
        
        print(f"Successfully generated perturbation dataset for {dataset_name} in: {output_dir}")
        print(f"Metrics saved to: {csv_path}")
        return str(output_dir), str(csv_path)


def get_parser():
    parser = argparse.ArgumentParser(description="生成医学2D图像扰动数据集")
    parser.add_argument("--output_root", default=None, help="扰动数据集根目录")
    parser.add_argument("--all_datasets", action="store_true", help="处理所有数据集")
    parser.add_argument("--dataset_name", type=str, default=None, help="要处理的数据集名称")
    parser.add_argument("--adaptive", action="store_true", help="启用自适应模式")
    parser.add_argument("--ssim_targets", type=str, default=None, help="SSIM目标范围（JSON字符串）")
    parser.add_argument("--calibration_samples", type=int, default=1, help="校准参数的图像数量（默认1）")
    parser.add_argument("--level_multipliers", type=str, default=None, 
                        help='Level强度乘数（JSON格式），例如: \'{"1": 2.0, "3": 3.0, "5": 4.0}\' 表示level1强度×2, level3强度×3, level5强度×4')
    return parser


def parse_ssim_targets(ssim_targets_str: str) -> Dict[int, Tuple[float, float]]:
    if ssim_targets_str is None:
        return None
    try:
        targets_dict = json.loads(ssim_targets_str)
        return {int(k): tuple(v) for k, v in targets_dict.items()}
    except Exception as e:
        raise ValueError(f"Invalid ssim_targets format: {e}")


def parse_level_multipliers(level_multipliers_str: str) -> Dict[int, float]:
    """
    解析level强度乘数参数
    
    Args:
        level_multipliers_str: JSON格式字符串，例如 '{"1": 2.0, "3": 3.0, "5": 4.0}'
    
    Returns:
        Dict[int, float]: level到乘数的映射
    """
    if level_multipliers_str is None:
        return None
    try:
        multipliers_dict = json.loads(level_multipliers_str)
        result = {int(k): float(v) for k, v in multipliers_dict.items()}
        
        # 验证level范围
        for level in result.keys():
            if level not in [1, 2, 3, 4, 5]:
                raise ValueError(f"Invalid level: {level}. Must be 1-5.")
        
        # 验证乘数范围
        for level, mult in result.items():
            if mult < 0.1 or mult > 10.0:
                raise ValueError(f"Multiplier for level {level} is {mult}, should be in range [0.1, 10.0]")
        
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Invalid level_multipliers format: {e}")


def main():
    args = get_parser().parse_args()
    
    datasets_config = {
        "isic_2016": {"name": "ISIC 2016", "resized_img_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/data/Segmentation_Data_2025/Part-1-Lesion-Segmentation-2016/Training/images_256", "image_extensions": ["jpg"]},
        "kelkalot-the-hyper-kvasir-dataset": {"name": "hyper-kvasir", "resized_img_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/data/Segmentation_Data_2025/kelkalot-the-hyper-kvasir-dataset/1/segmented-images/images_256/", "image_extensions": ["jpg"]},
        "nikhilroxtomar-brain-tumor": {"name": "brain-tumor", "resized_img_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/data/Segmentation_Data_2025/nikhilroxtomar-brain-tumor-segmentation/1/images_256/", "image_extensions": ["png"]},
        "aryashah2k-breast-ultrasound": {"name": "breast-ultrasound", "resized_img_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/data/Segmentation_Data_2025/aryashah2k-breast-ultrasound-images-dataset/Dataset_BUSI_with_GT/malignant_preprocessed/images_256/", "image_extensions": ["png"]},
        #"andrewmvd-cancer-inst": {"name": "cancer-inst", "resized_img_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/data/Segmentation_Data_2025/andrewmvd-cancer-inst-segmentation-and-classification/4/Part_1/images_256/", "image_extensions": ["png"]},
        "deathtrooper-multichannel-glaucoma-disc": {"name": "multichannel-glaucoma-disc", "resized_img_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/data/Segmentation_Data_2025/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/disc/images_256/", "image_extensions": ["png"]},
        "deathtrooper-multichannel-glaucoma-cup": {"name": "multichannel-glaucoma-cup", "resized_img_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/data/Segmentation_Data_2025/deathtrooper-multichannel-glaucoma-benchmark-dataset/10/organized_glaucoma_data/cup/images_256/", "image_extensions": ["png"]},
        #"prathamkumar0011-miccai-flare22-challenge": {"name": "miccai-flare22-challenge", "resized_img_dir": "/mnt/fast/nobackup/scratch4weeks/ly0008/cxx/data/Segmentation_Data_2025/prathamkumar0011-miccai-flare22-challenge-dataset/1/FLARE22Train/FLARE22Train/npy/CT_Abd/images_256/", "image_extensions": ["npy"], "npy_format": "single"}
    }
    
    perturbation_configs = {
        "isic_2016": {'base': ['gaussian_noise', 'gaussian_blur', 'compression_artifacts', 'brightness_contrast', 'pixelate'], 'modality_specific': ['dermoscopy_artifacts', 'light_reflection', 'hair_artifacts']},
        "kelkalot-the-hyper-kvasir-dataset": {'base': ['gaussian_noise', 'gaussian_blur', 'motion_blur', 'brightness_contrast'], 'modality_specific': ['specular_reflection', 'bubbles', 'blood', 'saturation']},
        "nikhilroxtomar-brain-tumor": {'base': ['gaussian_noise', 'gaussian_blur', 'motion_blur'], 'modality_specific': ['motion_artifacts', 'bias_field', 'ghosting']},
        "aryashah2k-breast-ultrasound": {'base': ['gaussian_noise', 'speckle_noise', 'gaussian_blur', 'motion_blur'], 'modality_specific': ['acoustic_shadowing', 'reverberation_artifacts']},
        "andrewmvd-cancer-inst": {'base': ['gaussian_noise', 'gaussian_blur', 'brightness_contrast', 'compression_artifacts'], 'modality_specific': ['stain_variation', 'bubble_artifacts']},
        "deathtrooper-multichannel-glaucoma-disc": {
            'base': ['gaussian_noise', 'gaussian_blur', 'motion_blur'],
            'modality_specific': ['shadow_artifacts', 'blink_artifacts', 'motion_artifacts_oct', 'defocus_artifacts', 'speckle_artifacts']
        },
        "deathtrooper-multichannel-glaucoma-cup": {
            'base': ['gaussian_noise', 'gaussian_blur', 'motion_blur'],
            'modality_specific': ['shadow_artifacts', 'blink_artifacts', 'motion_artifacts_oct', 'defocus_artifacts', 'speckle_artifacts']
        },
        "prathamkumar0011-miccai-flare22-challenge": {'base': ['gaussian_noise', 'gaussian_blur', 'motion_blur'], 'modality_specific': ['metal_artifacts', 'beam_hardening', 'photon_starvation']}
    }
    
    # 解析level强度乘数
    level_multipliers = parse_level_multipliers(args.level_multipliers)
    
    # 创建扰动生成器，传入level_multipliers
    pert_generator = Medical2DPerturbationGenerator(seed=42, level_multipliers=level_multipliers)
    ssim_targets = parse_ssim_targets(args.ssim_targets) if args.adaptive else None
    
    # 打印配置信息
    if level_multipliers:
        print(f"🎚️  Level强度乘数:")
        for level in sorted(level_multipliers.keys()):
            mult = level_multipliers[level]
            print(f"   Level {level}: ×{mult:.1f}")
        print()
    
    if args.adaptive:
        if ssim_targets is None:
            ssim_targets = pert_generator.default_ssim_targets
        print(f"🎯 Adaptive mode enabled with SSIM targets:")
        for level, (min_s, max_s) in ssim_targets.items():
            print(f"   Level {level}: [{min_s:.2f}, {max_s:.2f}]")
        print(f"📊 Calibration samples: {args.calibration_samples}")
    
    if args.all_datasets:
        datasets_to_process = datasets_config.keys()
    elif args.dataset_name:
        if args.dataset_name in datasets_config:
            datasets_to_process = [args.dataset_name]
        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")
    else:
        raise ValueError("Please specify --all_datasets or --dataset_name")
    
    output_dirs, csv_files, skipped_datasets = {}, {}, []
    
    for dataset_name in datasets_to_process:
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_config = datasets_config[dataset_name]
        perturbation_config = perturbation_configs[dataset_name]
        img_dir = dataset_config['resized_img_dir']
        if not os.path.exists(img_dir):
            print(f"⚠️  SKIPPING: Directory does not exist: {img_dir}")
            skipped_datasets.append(dataset_name)
            continue
        try:
            output_dir, csv_path = pert_generator.generate_perturbation_dataset(
                dataset_config, perturbation_config, args.output_root,
                adaptive_mode=args.adaptive, ssim_targets=ssim_targets,
                calibration_samples=args.calibration_samples)
            output_dirs[dataset_name] = output_dir
            csv_files[dataset_name] = csv_path
        except Exception as e:
            print(f"❌ ERROR processing {dataset_name}: {e}")
            skipped_datasets.append(dataset_name)
    
    print("\n" + "="*60)
    print("All perturbation datasets generated!")
    print("="*60)
    if output_dirs:
        print("\n✅ Successfully processed datasets:")
        for dataset_name, output_dir in output_dirs.items():
            print(f"  • {dataset_name}: {output_dir}")
    if skipped_datasets:
        print(f"\n⚠️  Skipped {len(skipped_datasets)} datasets")
    print(f"\n📊 Summary: {len(output_dirs)} processed, {len(skipped_datasets)} skipped")


if __name__ == "__main__":
    main()
