# models/sam2_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 导入本地创建的模块
try:
    from .sam2_core.hieradet import Hiera, FpnNeck
except ImportError:
    # 兼容直接运行此文件的情况
    from models.sam2_core.hieradet import Hiera, FpnNeck

class SAM2Wrapper(nn.Module):
    """SAM2模型封装，使用本地代码加载预训练权重"""
    
    def __init__(self, checkpoint_path: str, model_cfg: str, 
                 config_dir: str = None, freeze_encoder: bool = True):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.freeze_encoder = freeze_encoder
        
        # 1. 实例化模型 (硬编码 Tiny 配置，与 sam2.1_hiera_t.yaml 一致)
        # Hiera Tiny Config
        self.trunk = Hiera(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 14, 7]
        )
        
        # FPN Neck Config
        self.neck = FpnNeck(
            d_model=256,
            backbone_channel_list=[768, 384, 192, 96], # Neck convs[0] 对应 768
            fpn_top_down_levels=[2, 3] 
        )
        
        # 2. 加载权重
        self._load_weights()
        
        if self.freeze_encoder:
            self._freeze_image_encoder()
            
        # 记录输出维度
        self.encoder_embed_dim = 256

    def _load_weights(self):
        if not os.path.exists(self.checkpoint_path):
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}, using random init.")
            return

        print(f"Loading SAM2 checkpoint from {self.checkpoint_path}...")
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        
        # 处理可能的 'model' 键
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        # 过滤并加载 trunk (backbone) 权重
        trunk_dict = {}
        neck_dict = {}
        
        for k, v in state_dict.items():
            # 官方权重前缀通常是 image_encoder.trunk.xxx
            if k.startswith("image_encoder.trunk."):
                new_k = k.replace("image_encoder.trunk.", "")
                trunk_dict[new_k] = v
            # 官方权重前缀通常是 image_encoder.neck.xxx
            elif k.startswith("image_encoder.neck."):
                new_k = k.replace("image_encoder.neck.", "")
                neck_dict[new_k] = v
                
        # 加载
        if trunk_dict:
            msg = self.trunk.load_state_dict(trunk_dict, strict=False)
            print(f"Trunk loaded: {msg}")
        if neck_dict:
            # 现在 FpnNeck 结构已修复，应该能完全匹配
            msg = self.neck.load_state_dict(neck_dict, strict=False)
            print(f"Neck loaded: {msg}")
            
    def _freeze_image_encoder(self):
        for param in self.trunk.parameters():
            param.requires_grad = False
        for param in self.neck.parameters():
            param.requires_grad = False
        print("Image encoder (trunk + neck) frozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hiera backbone forward
        # 输出是 [feat_96, feat_192, feat_384, feat_768] (从浅到深)
        backbone_out = self.trunk(x)
        
        # 错误修复：不要反转 backbone_out
        # FpnNeck 的 forward 循环 for i in range(3, -1, -1) 会首先取 xs[3] (即 768 通道特征)
        # 并将其送入 convs[0] (接收 768 通道)
        # 所以保持 Hiera 的原始输出顺序是正确的
        
        features = self.neck(backbone_out)
        
        # Neck 输出顺序与 backbone_out 顺序一致 (即 [浅...深])
        # features[0] 对应 level 0 (high res, stride 4, 256 channels)
        # 实际上 FpnNeck 的 out[i] 对应输入 xs[i] 的层级
        # 所以 features[0] 是对应 xs[0] (stride 4) 的输出
        
        # 我们返回最高分辨率特征 (stride 4)
        return features[0]

    def get_encoder_output_dim(self) -> int:
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
        # SAM2输出通常是输入的1/4 (stride 4)，即256x256
        # 如果目标是 1/16 (stride 16)，我们需要下采样
        # 这里的逻辑根据你的具体需求调整，目前保留原逻辑
        target_feat_size = self.target_size // 4 # 假设我们想要 stride 4 的特征
        
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