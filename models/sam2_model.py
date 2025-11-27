import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import os
import sys

class SAM2Wrapper(nn.Module):
    """SAM2模型封装，用于加载预训练权重"""
    
    def __init__(self, checkpoint_path: str, model_cfg: str, 
                 config_dir: str = None, freeze_encoder: bool = True):
        """
        Args:
            checkpoint_path: SAM2预训练权重路径
            model_cfg: SAM2模型配置文件名 (如 "sam2_hiera_l.yaml")
            config_dir: SAM2配置文件所在目录
            freeze_encoder: 是否冻结image encoder
        """
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.config_dir = config_dir
        self.freeze_encoder = freeze_encoder
        
        # 加载SAM2模型
        self._load_sam2_model()
        
        if self.freeze_encoder:
            self._freeze_image_encoder()
    
    def _load_sam2_model(self):
        """加载SAM2预训练模型"""
        try:
            # 添加SAM2路径到sys.path
            sam2_path = "segment-anything-2"
            if os.path.exists(sam2_path) and sam2_path not in sys.path:
                sys.path.insert(0, sam2_path)
            
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # 构建配置文件完整路径
            if self.config_dir:
                config_file = os.path.join(self.config_dir, self.model_cfg)
            else:
                config_file = self.model_cfg
            
            print(f"Loading SAM2 model from config: {config_file}")
            print(f"Loading SAM2 checkpoint: {self.checkpoint_path}")
            
            # 构建SAM2模型
            self.sam2_model = build_sam2(
                config_file=config_file,
                ckpt_path=self.checkpoint_path,
                device='cpu'  # 先在CPU上加载
            )
            
            # 提取各个组件
            self.image_encoder = self.sam2_model.image_encoder
            self.prompt_encoder = self.sam2_model.sam_prompt_encoder
            self.mask_decoder = self.sam2_model.sam_mask_decoder
            
            # 获取编码器输出维度
            self.encoder_embed_dim = self.sam2_model.image_encoder.neck[0].out_channels
            
            print(f"SAM2 model loaded successfully!")
            print(f"Image encoder output dimension: {self.encoder_embed_dim}")
            
        except ImportError as e:
            print(f"Warning: SAM2 not found ({e}), using dummy encoder")
            self._create_dummy_encoder()
        except Exception as e:
            print(f"Error loading SAM2: {e}")
            print("Using dummy encoder for testing")
            self._create_dummy_encoder()
    
    def _create_dummy_encoder(self):
        """创建简化的编码器用于测试"""
        print("Creating dummy encoder for testing...")
        
        # 简化的编码器
        self.image_encoder = nn.Sequential(
            # Stage 1: 256x256 -> 128x128
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Stage 2: 128x128 -> 64x64
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 3: 64x64 -> 32x32
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 4: 32x32 -> 16x16
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.prompt_encoder = None
        self.mask_decoder = None
        self.encoder_embed_dim = 256
        
        print(f"Dummy encoder created with output dim: {self.encoder_embed_dim}")
    
    def _freeze_image_encoder(self):
        """冻结image encoder"""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        print("Image encoder frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            features: 编码特征 [B, C, H', W']
        """
        # 如果使用真实SAM2，需要调整输入尺寸
        original_size = x.shape[2:]
        
        with torch.set_grad_enabled(not self.freeze_encoder):
            features = self.image_encoder(x)
        
        return features
    
    def get_encoder_output_dim(self) -> int:
        """获取编码器输出维度"""
        return self.encoder_embed_dim


class SAM2EncoderWithResize(nn.Module):
    """
    SAM2编码器包装器，处理输入尺寸调整
    SAM2期望1024x1024输入，但我们可能使用256x256训练
    """
    
    def __init__(self, sam2_wrapper: SAM2Wrapper, target_size: int = 256):
        super().__init__()
        self.sam2_wrapper = sam2_wrapper
        self.target_size = target_size
        self.sam2_input_size = 1024  # SAM2标准输入尺寸
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, target_size, target_size]
        Returns:
            features: [B, C, H', W']
        """
        B, C, H, W = x.shape
        
        # 如果输入尺寸不是SAM2期望的尺寸，需要调整
        if H != self.sam2_input_size or W != self.sam2_input_size:
            # 上采样到SAM2输入尺寸
            x_resized = F.interpolate(
                x, 
                size=(self.sam2_input_size, self.sam2_input_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            x_resized = x
        
        # 通过SAM2编码器
        features = self.sam2_wrapper(x_resized)
        
        # 下采样特征到合适的尺寸
        # SAM2输出通常是输入的1/16，即64x64
        # 我们需要调整到与target_size匹配的尺寸
        target_feat_size = self.target_size // 16  # 假设16倍下采样
        
        if features.shape[2] != target_feat_size:
            features = F.interpolate(
                features,
                size=(target_feat_size, target_feat_size),
                mode='bilinear',
                align_corners=False
            )
        
        return features
    
    def get_encoder_output_dim(self) -> int:
        return self.sam2_wrapper.get_encoder_output_dim()
