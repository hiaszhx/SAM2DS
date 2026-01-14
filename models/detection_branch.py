# models/detection_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. RDIAN 风格的组件 (适配版)
# ==============================================================================

class ChannelGate(nn.Module):
    """RDIAN 中的通道注意力 (简化版，移除复杂的池化选项，保留核心 Avg+Max)"""
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        # Avg Pool
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_avg = self.mlp(avg_pool)
        
        # Max Pool
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_max = self.mlp(max_pool)
        
        # Sum & Sigmoid
        channel_att_sum = channel_att_avg + channel_att_max
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    """RDIAN 中的空间注意力"""
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, bias=False)

    def forward(self, x):
        # Channel Pool: Max & Avg
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) 
        return x * scale

class CBAM(nn.Module):
    """完整的 CBAM 模块"""
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class RDIANBlock(nn.Module):
    """
    适配 RDIAN 的 NewBlock。
    结构: Conv(in->in/2) -> BN -> ReLU -> Conv(in/2->in) -> BN -> ReLU + Residual
    关键修复: LeakyReLU 设置为 inplace=False，避免梯度计算冲突
    """
    def __init__(self, in_channels, kernel_size, padding):
        super(RDIANBlock, self).__init__()
        reduced_channels = max(8, in_channels // 2) # 确保通道数不因过小而报错
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size, padding=padding, stride=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.LeakyReLU(inplace=False) # FIX: Changed to False
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(reduced_channels, in_channels, kernel_size, padding=padding, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=False) # FIX: Changed to False to allow out += residual
        )

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out

# ==============================================================================
# 2. 核心组件 (保持原 SAM2DS 逻辑)
# ==============================================================================

class SoftArgmax2D(nn.Module):
    """
    可微 Soft-Argmax 模块：从热力图中直接回归坐标
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
        x_coord = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
        y_coord = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        
        # 3. 计算重心 (期望值)
        pos_x = (heatmap * x_coord).sum(dim=(2, 3)) # [B, 1]
        pos_y = (heatmap * y_coord).sum(dim=(2, 3)) # [B, 1]
        
        # [B, 2] -> (x, y)
        return torch.cat([pos_x, pos_y], dim=1)

# ==============================================================================
# 3. 新的 Detection Branch 类
# ==============================================================================

class DetectionBranch(nn.Module):
    """
    RDIAN-Style 探测分支:
    1. Adapter: 降维到 hidden_dim (e.g., 64)
    2. Parallel Multi-Scale Blocks: 1x1, 3x3, 5x5, 7x7 并行处理
    3. Fusion: Concat -> Conv -> CBAM
    4. Head: 生成 Heatmap
    5. SoftArgmax: 生成坐标
    """
    
    def __init__(self, in_channels: int, hidden_dim: int = 64, adapter_dim: int = 256):
        super().__init__()
        
        # 强制使用较小的内部维度以模拟 RDIAN 的轻量级设计
        internal_dim = 64 
        
        # 1. Adapter层: 调整通道数并降维，准备进入 RDIAN 模块
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, internal_dim, 1),
            nn.BatchNorm2d(internal_dim),
            nn.ReLU(inplace=True),
        )
        
        # 2. 多尺度并行分支 (仿照 RDIAN 的 residual_block 0~3)
        self.block1 = RDIANBlock(internal_dim, kernel_size=1, padding=0)
        self.block3 = RDIANBlock(internal_dim, kernel_size=3, padding=1)
        self.block5 = RDIANBlock(internal_dim, kernel_size=5, padding=2)
        self.block7 = RDIANBlock(internal_dim, kernel_size=7, padding=3)
        
        # 3. 融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(internal_dim * 4, internal_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(internal_dim),
            nn.LeakyReLU(inplace=True) # 这里没有 Residual 连接，可以使用 inplace=True
        )
        
        # 4. CBAM 注意力
        self.cbam = CBAM(internal_dim)
        
        # 5. 热力图预测头
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(internal_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1) # 输出单通道 logits
        )
        
        # 6. 坐标回归
        self.soft_argmax = SoftArgmax2D(beta=150.0)
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: 编码器特征 [B, C, H, W]
        """
        # 1. Adapter & 降维
        feat = self.adapter(x) # [B, 64, H, W]
        
        # 2. 并行多尺度特征提取
        f1 = self.block1(feat)
        f3 = self.block3(feat)
        f5 = self.block5(feat)
        f7 = self.block7(feat)
        
        # 3. 拼接与融合
        cat_feat = torch.cat([f1, f3, f5, f7], dim=1)
        fused_feat = self.fusion_conv(cat_feat)
        
        # 4. CBAM 增强
        refined_feat = self.cbam(fused_feat)
        
        # 5. 生成热力图
        heatmap_logits = self.heatmap_head(refined_feat)
        
        # 6. 回归坐标
        coords = self.soft_argmax(heatmap_logits)
        
        return {
            'heatmap': heatmap_logits,
            'coords': coords,
            'features': refined_feat 
        }