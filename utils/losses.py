import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    """Dice Loss"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class DetectionLoss(nn.Module):
    """探测分支损失"""
    
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: dict, gt_mask: torch.Tensor, gt_coords: torch.Tensor) -> dict:
        """
        Args:
            predictions: 探测分支输出
            gt_mask: GT掩码 [B, 1, H, W]
            gt_coords: GT坐标 [B, 2]
        """
        # 热力图损失
        heatmap = predictions['heatmap']
        gt_heatmap = F.interpolate(gt_mask, size=heatmap.shape[2:], mode='bilinear', align_corners=False)
        heatmap_loss = self.focal_loss(heatmap, gt_heatmap)
        
        # 坐标回归损失
        pred_coords = predictions['coords']
        coord_loss = self.mse_loss(pred_coords, gt_coords)
        
        total_loss = heatmap_loss + coord_loss
        
        return {
            'total': total_loss,
            'heatmap': heatmap_loss,
            'coord': coord_loss
        }

class SegmentationLoss(nn.Module):
    """分割分支损失"""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            predictions: 预测掩码 [B, 1, H, W]
            targets: GT掩码 [B, 1, H, W]
        """
        # 调整target大小
        if predictions.shape != targets.shape:
            targets = F.interpolate(targets, size=predictions.shape[2:], mode='bilinear', align_corners=False)
        
        dice_loss = self.dice_loss(predictions, targets)
        bce_loss = self.bce_loss(predictions, targets)
        
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return {
            'total': total_loss,
            'dice': dice_loss,
            'bce': bce_loss
        }

class CombinedLoss(nn.Module):
    """组合损失"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.detection_loss = DetectionLoss(
            focal_alpha=config['loss']['focal_alpha'],
            focal_gamma=config['loss']['focal_gamma']
        )
        self.segmentation_loss = SegmentationLoss(
            dice_weight=config['loss']['dice_weight'],
            bce_weight=config['loss']['bce_weight']
        )
        self.seg_weight = config['loss']['seg_loss_weight']
        self.det_weight = config['loss']['det_loss_weight']
    
    def forward(self, predictions: dict, gt_mask: torch.Tensor, gt_coords: torch.Tensor) -> dict:
        """
        Args:
            predictions: 模型输出
            gt_mask: GT掩码
            gt_coords: GT坐标
        """
        det_losses = self.detection_loss(predictions['detection'], gt_mask, gt_coords)
        seg_losses = self.segmentation_loss(predictions['segmentation'], gt_mask)
        
        total_loss = self.det_weight * det_losses['total'] + self.seg_weight * seg_losses['total']
        
        return {
            'total': total_loss,
            'detection': det_losses,
            'segmentation': seg_losses
        }
