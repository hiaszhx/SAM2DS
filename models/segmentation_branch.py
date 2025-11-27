import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationBranch(nn.Module):
    """分割分支"""
    
    def __init__(self, in_channels: int, adapter_dim: int = 256):
        super().__init__()
        
        # Adapter层
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, adapter_dim, 1),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(adapter_dim, adapter_dim, 3, padding=1),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True)
        )
        
        # Prompt融合模块
        self.prompt_fusion = nn.Sequential(
            nn.Conv2d(adapter_dim + 64, adapter_dim, 3, padding=1),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.ModuleList([
            self._make_decoder_block(adapter_dim, 128),
            self._make_decoder_block(128, 64),
            self._make_decoder_block(64, 32),
        ])
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )
        
    def _make_decoder_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def encode_point_prompt(self, coords: torch.Tensor, feature_size: tuple) -> torch.Tensor:
        """
        将点坐标编码为特征图
        Args:
            coords: [B, 2] 归一化坐标 (x, y)
            feature_size: (H, W) 特征图尺寸
        Returns:
            prompt_features: [B, 64, H, W]
        """
        B = coords.shape[0]
        H, W = feature_size
        device = coords.device
        
        # 创建高斯热力图
        y_coords = torch.arange(H, device=device).float() / H
        x_coords = torch.arange(W, device=device).float() / W
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1, 2, H, W]
        
        # 计算距离
        coords_expanded = coords.view(B, 2, 1, 1)  # [B, 2, 1, 1]
        dist = torch.sum((grid - coords_expanded) ** 2, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 高斯核
        sigma = 0.1
        gaussian = torch.exp(-dist / (2 * sigma ** 2))
        
        # 扩展通道
        prompt_features = gaussian.repeat(1, 64, 1, 1)
        
        return prompt_features
    
    def forward(self, x: torch.Tensor, prompt_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 编码器特征 [B, C, H, W]
            prompt_coords: 点坐标 [B, 2]
        Returns:
            mask: 分割掩码 [B, 1, H', W']
        """
        # Adapter
        adapted_features = self.adapter(x)
        B, C, H, W = adapted_features.shape
        
        # 编码prompt
        prompt_features = self.encode_point_prompt(prompt_coords, (H, W))
        
        # 融合prompt和特征
        fused_features = torch.cat([adapted_features, prompt_features], dim=1)
        fused_features = self.prompt_fusion(fused_features)
        
        # 解码
        x = fused_features
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        # 输出
        mask = self.output_head(x)
        
        return mask
