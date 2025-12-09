# models/detection_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftArgmax2D(nn.Module):
    """
    可微 Soft-Argmax 模块：从热力图中直接回归坐标
    参数 beta 控制 softmax 的尖锐程度，beta 越大越接近 argmax
    """
    def __init__(self, beta=100.0):
        super().__init__()
        self.beta = beta

    def forward(self, heatmap):
        # heatmap: [B, 1, H, W]
        B, C, H, W = heatmap.shape
        device = heatmap.device
        
        # 1. 计算空间 Softmax
        heatmap = heatmap.view(B, -1)
        heatmap = F.softmax(self.beta * heatmap, dim=1)
        heatmap = heatmap.view(B, C, H, W)
        
        # 2. 生成归一化坐标网格
        # x: [0, 1/W, ..., 1]
        x_coord = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
        # y: [0, 1/H, ..., 1]
        y_coord = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        
        # 3. 计算重心 (期望值)
        # 积分: sum(P(x,y) * x)
        pos_x = (heatmap * x_coord).sum(dim=(2, 3)) # [B, 1]
        pos_y = (heatmap * y_coord).sum(dim=(2, 3)) # [B, 1]
        
        # [B, 2] -> (x, y)
        return torch.cat([pos_x, pos_y], dim=1)

class ResBlock(nn.Module):
    """简单的残差卷积块，用于增加深度而不丢失梯度"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class DetectionBranch(nn.Module):
    """
    增强型探测分支：
    1. Adapter 调整维度
    2. Deep Feature Extractor (ResBlocks) 提取深层特征
    3. Heatmap Head 预测概率图
    4. Soft-Argmax 回归坐标
    """
    
    def __init__(self, in_channels: int, hidden_dim: int = 256, adapter_dim: int = 256):
        super().__init__()
        
        # 1. Adapter层 (保持不变)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, adapter_dim, 1),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True),
        )
        
        # 2. 增强特征提取 (加深网络，增加参数量)
        # 使用 3 个残差块，通道数增加到 256，增强上下文建模能力
        self.feature_extractor = nn.Sequential(
            ResBlock(adapter_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
            ResBlock(hidden_dim, hidden_dim),
        )
        
        # 3. 热力图预测头
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1) # 输出单通道 logits
        )
        
        # 4. 坐标回归 (无参数，纯数学计算)
        self.soft_argmax = SoftArgmax2D(beta=150.0) # 稍微调大beta，让热点更聚焦
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: 编码器特征 [B, C, H, W]
        Returns:
            dict: {
                'heatmap': 热力图 logits [B, 1, H, W],
                'coords': 归一化坐标 [B, 2],
                'features': 适配后的特征 [B, adapter_dim, H, W]
            }
        """
        # Adapter
        adapted_features = self.adapter(x)
        
        # Deep Feature Extraction
        features = self.feature_extractor(adapted_features)
        
        # Heatmap Prediction
        heatmap_logits = self.heatmap_head(features)
        
        # Soft-Argmax Coordinate Regression
        # 注意：这里对 heatmap_logits 做 soft-argmax，内部会自动做 softmax
        coords = self.soft_argmax(heatmap_logits)
        
        return {
            'heatmap': heatmap_logits,
            'coords': coords,
            'features': adapted_features
        }