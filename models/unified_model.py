# models/unified_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sam2_model import SAM2Wrapper
from .detection_branch import DetectionBranch
from .segmentation_branch import SegmentationBranch

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享 MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道方向做平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """CBAM 混合注意力模块"""
    def __init__(self, planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class FeatureFusion(nn.Module):
    """
    带注意力机制的特征融合模块：
    1. 自顶向下融合 (Top-Down Fusion)
    2. CBAM 注意力过滤背景噪声
    """
    def __init__(self, in_channels=256):
        super().__init__()
        
        # 融合后的平滑卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # === 新增：CBAM 注意力模块 ===
        self.attention = CBAM(in_channels)

    def forward(self, features):
        # features: [s4, s8, s16, s32]
        # 从最深层开始
        fused = features[-1] # s32
        
        # 自顶向下逐层上采样并相加
        for i in range(len(features) - 2, -1, -1):
            fused = F.interpolate(fused, scale_factor=2, mode='bilinear', align_corners=False)
            target_feat = features[i]
            
            # 处理可能的尺寸不匹配 (padding)
            if fused.shape[-2:] != target_feat.shape[-2:]:
                fused = F.interpolate(fused, size=target_feat.shape[-2:], mode='bilinear', align_corners=False)
            
            fused = fused + target_feat
            
        # 卷积平滑
        out = self.fusion_conv(fused)
        
        # === 应用注意力机制 ===
        # 这一步非常关键，它会根据融合后的特征自动抑制背景（Fa降低），增强目标（Pd提升）
        out = self.attention(out)
        
        return out

class UnifiedModel(nn.Module):
    """
    Unified Model with Dual Necks, Dual Fusion, and Attention
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
        
        # 2. 初始化两个独立的特征融合模块 (现在包含了 CBAM)
        self.fusion_det = FeatureFusion(in_channels=encoder_dim)
        self.fusion_seg = FeatureFusion(in_channels=encoder_dim)
        
        # 3. 探测分支 (现在包含了 Deep ResBlock 和 Soft-Argmax)
        self.detection_branch = DetectionBranch(
            in_channels=encoder_dim,
            hidden_dim=config['model']['detection_hidden_dim'], # 建议在 config 中设为 256 或 512
            adapter_dim=config['model']['adapter_dim']
        )
        
        # 4. 分割分支
        self.segmentation_branch = SegmentationBranch(
            in_channels=encoder_dim,
            adapter_dim=config['model']['adapter_dim']
        )
        
        print("Unified model initialized with Enhanced Detection Head (SoftArgmax) and CBAM Fusion.")
    
    def forward(self, images: torch.Tensor, prompt_coords: torch.Tensor = None, 
                use_detection_prompt: bool = False) -> dict:
        
        input_shape = images.shape[-2:]

        # 1. 编码 (获取两组多尺度特征)
        features_det_list, features_seg_list = self.encoder(images)
        
        # 2. 独立融合 (带注意力)
        fused_det = self.fusion_det(features_det_list) # 给探测分支用
        fused_seg = self.fusion_seg(features_seg_list) # 给分割分支用
        
        # 3. 探测分支 (使用 fused_det)
        detection_output = self.detection_branch(fused_det)
        
        # 4. 确定 Prompt
        if use_detection_prompt:
            # 这里的 coords 已经是 Soft-Argmax 出来的精确坐标了
            prompt_coords_used = detection_output['coords']
            # 如果是推理阶段，你可能想要 detach，但在训练阶段如果想训练 detection_branch 让分割更好，
            # 可以保留梯度 (取决于你的 CombinedLoss 策略，通常建议 detach 以独立优化检测头)
            if not self.training:
                prompt_coords_used = prompt_coords_used.detach()
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