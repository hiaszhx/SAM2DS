import torch
import numpy as np
from sklearn.metrics import precision_recall_curve

def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算IoU"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()

def compute_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """计算F1 Score"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return f1.item()

def compute_pd_fa(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple:
    """计算Pd (检测率) 和 Fa (虚警率)"""
    pred = torch.sigmoid(pred).cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    pred_binary = (pred > threshold).astype(np.float32)
    
    # Pd: 检测率
    tp = np.sum(pred_binary * target)
    fn = np.sum((1 - pred_binary) * target)
    pd = tp / (tp + fn + 1e-6)
    
    # Fa: 虚警率 (每百万像素)
    fp = np.sum(pred_binary * (1 - target))
    total_pixels = target.size
    fa = fp / total_pixels * 1e6
    
    return pd, fa

class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.iou_sum = 0
        self.f1_sum = 0
        self.pd_sum = 0
        self.fa_sum = 0
        self.count = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        iou = compute_iou(pred, target)
        f1 = compute_f1(pred, target)
        pd, fa = compute_pd_fa(pred, target)
        
        self.iou_sum += iou
        self.f1_sum += f1
        self.pd_sum += pd
        self.fa_sum += fa
        self.count += 1
    
    def compute(self) -> dict:
        return {
            'IoU': self.iou_sum / self.count * 100,
            'F1': self.f1_sum / self.count * 100,
            'Pd': self.pd_sum / self.count * 100,
            'Fa': self.fa_sum / self.count
        }
