import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionBranch(nn.Module):
    """探测分支 - 输出单点坐标"""
    
    def __init__(self, in_channels: int, hidden_dim: int = 512, adapter_dim: int = 256):
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
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(adapter_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 热力图预测
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )
        
        # 全局平均池化 + 坐标回归
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.coord_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),  # (x, y) 坐标
            nn.Sigmoid()  # 归一化到[0, 1]
        )
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: 编码器特征 [B, C, H, W]
        Returns:
            dict: {
                'heatmap': 热力图 [B, 1, H, W],
                'coords': 归一化坐标 [B, 2],
                'features': 适配后的特征 [B, adapter_dim, H, W]
            }
        """
        # Adapter
        adapted_features = self.adapter(x)
        
        # 特征提取
        features = self.feature_extractor(adapted_features)
        
        # 热力图预测
        heatmap = self.heatmap_head(features)
        
        # 坐标回归
        pooled = self.global_pool(features).flatten(1)
        coords = self.coord_regressor(pooled)
        
        return {
            'heatmap': heatmap,
            'coords': coords,
            'features': adapted_features
        }
    
    def get_point_from_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """从热力图中提取最大响应点坐标"""
        B, _, H, W = heatmap.shape
        heatmap_flat = heatmap.view(B, -1)
        max_indices = torch.argmax(heatmap_flat, dim=1)
        
        y_coords = (max_indices // W).float() / H
        x_coords = (max_indices % W).float() / W
        
        coords = torch.stack([x_coords, y_coords], dim=1)
        return coords
