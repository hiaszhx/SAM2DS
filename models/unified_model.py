# models/unified_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sam2_model import SAM2Wrapper
from .detection_branch import DetectionBranch
from .segmentation_branch import SegmentationBranch

class FeatureFusion(nn.Module):
    """
    轻量级特征融合模块：将 SAM2 的多尺度特征自顶向下融合。
    """
    def __init__(self, in_channels=256):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        # features: [s4, s8, s16, s32]
        fused = features[-1] # s32
        
        for i in range(len(features) - 2, -1, -1):
            fused = F.interpolate(fused, scale_factor=2, mode='bilinear', align_corners=False)
            target_feat = features[i]
            if fused.shape[-2:] != target_feat.shape[-2:]:
                fused = F.interpolate(fused, size=target_feat.shape[-2:], mode='bilinear', align_corners=False)
            fused = fused + target_feat
            
        out = self.fusion_conv(fused)
        return out

class UnifiedModel(nn.Module):
    """
    Unified Model with Dual Necks and Dual Fusion
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # 1. 初始化双 Neck 编码器
        self.encoder = SAM2Wrapper(
            checkpoint_path=config['model']['sam2_checkpoint'],
            model_cfg=config['model']['sam2_model_cfg'],
            config_dir=config['model'].get('sam2_config_dir', None),
            freeze_trunk=config['model'].get('freeze_trunk', True),
            freeze_neck_det=config['model'].get('freeze_neck_det', False),
            freeze_neck_seg=config['model'].get('freeze_neck_seg', False)
        )
        
        encoder_dim = self.encoder.get_encoder_output_dim()
        
        # 2. 初始化两个独立的特征融合模块
        self.fusion_det = FeatureFusion(in_channels=encoder_dim)
        self.fusion_seg = FeatureFusion(in_channels=encoder_dim)
        
        # 3. 探测分支
        self.detection_branch = DetectionBranch(
            in_channels=encoder_dim,
            hidden_dim=config['model']['detection_hidden_dim'],
            adapter_dim=config['model']['adapter_dim']
        )
        
        # 4. 分割分支
        self.segmentation_branch = SegmentationBranch(
            in_channels=encoder_dim,
            adapter_dim=config['model']['adapter_dim']
        )
        
        print("Unified model initialized with Dual Necks and Dual Fusion paths.")
    
    def forward(self, images: torch.Tensor, prompt_coords: torch.Tensor = None, 
                use_detection_prompt: bool = False) -> dict:
        
        input_shape = images.shape[-2:]

        # 1. 编码 (获取两组多尺度特征)
        features_det_list, features_seg_list = self.encoder(images)
        
        # 2. 独立融合
        fused_det = self.fusion_det(features_det_list) # 给探测分支用
        fused_seg = self.fusion_seg(features_seg_list) # 给分割分支用
        
        # 3. 探测分支 (使用 fused_det)
        detection_output = self.detection_branch(fused_det)
        
        # 4. 确定 Prompt
        if use_detection_prompt:
            prompt_coords_used = detection_output['coords'].detach()
        else:
            assert prompt_coords is not None
            prompt_coords_used = prompt_coords
        
        # 5. 分割分支 (使用 fused_seg 和 prompt)
        segmentation_output = self.segmentation_branch(fused_seg, prompt_coords_used)
        
        # 尺寸恢复
        if segmentation_output.shape[-2:] != input_shape:
            segmentation_output = F.interpolate(
                segmentation_output, 
                size=input_shape, 
                mode='bilinear', 
                align_corners=False
            )
        
        return {
            'detection': detection_output,
            'segmentation': segmentation_output,
            'prompt_coords': prompt_coords_used
        }