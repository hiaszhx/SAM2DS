import os
import yaml
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.unified_model import UnifiedModel
from irstd_dataset import IRSTDDataset
from utils.losses import CombinedLoss
from utils.metrics import MetricsCalculator
# 引入 Adan 优化器 (需确保 utils/adan.py 存在)
from utils.adan import Adan

def setup_logging(log_file):
    """设置日志，同时输出到控制台和文件"""
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s') # 简化控制台输出，去掉时间前缀，因为训练代码里自己打印了
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    # 文件日志保留详细时间戳
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    return logger

class Trainer:
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_save_dir = self.config['logging']['save_dir']
        self.save_dir = os.path.join(base_save_dir, run_timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        log_path = os.path.join(self.save_dir, 'train_log.log')
        self.logger = setup_logging(log_path)
        
        self.logger.info(f"Training Run ID: {run_timestamp}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # 创建模型
        self.model = UnifiedModel(self.config).to(self.device)
        
        # 参数统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model Params: {total_params/1e6:.2f}M (Trainable: {trainable_params/1e6:.2f}M)")
        
        # 数据集
        self.train_dataset = IRSTDDataset(
            root_dir=self.config['dataset']['root_dir'],
            split=self.config['dataset']['train_split'],
            image_size=self.config['model']['input_size']
        )
        self.val_dataset = IRSTDDataset(
            root_dir=self.config['dataset']['root_dir'],
            split=self.config['dataset']['val_split'],
            image_size=self.config['model']['input_size']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        
        self.criterion = CombinedLoss(self.config).to(self.device)
        
        # 优化器设置
        opt_type = self.config['optimizer']['type']
        # 【关键修改】：强制转换为 float
        lr = float(self.config['training']['learning_rate'])
        weight_decay = float(self.config['training']['weight_decay'])

        if opt_type == 'Adan':
            self.optimizer = Adan(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr, 
                weight_decay=weight_decay,
                betas=self.config['optimizer']['betas']
            )
            self.logger.info("Using Adan Optimizer")
        else:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=weight_decay
            )
            self.logger.info("Using AdamW Optimizer")
        
        # 学习率调度器
        # 【关键修改】：强制转换 eta_min 为 float
        eta_min = float(self.config['scheduler']['eta_min'])
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=eta_min 
        )
        
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))
        self.best_iou = 0.0

    def get_prompt_strategy(self, epoch: int) -> float:
        schedule = self.config['training']['gt_prompt_ratio_schedule']
        start_epoch = schedule.get('start_epoch', 0)
        end_epoch = schedule.get('end_epoch', 10)
        start_ratio = schedule.get('start_ratio', 1.0)
        end_ratio = schedule.get('end_ratio', 0.0)
        
        if epoch < start_epoch:
            return float(start_ratio)
        elif epoch >= end_epoch:
            return float(end_ratio)
        else:
            total_steps = end_epoch - start_epoch
            current_step = epoch - start_epoch
            progress = current_step / total_steps
            ratio_diff = start_ratio - end_ratio
            gt_ratio = start_ratio - (progress * ratio_diff)
            return float(max(0.0, min(1.0, gt_ratio)))

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss_sum = 0.0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        
        gt_prompt_ratio = self.get_prompt_strategy(epoch)
        det_prompt_ratio = 1.0 - gt_prompt_ratio  
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["num_epochs"]}', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            coords = batch['coords'].to(self.device)
            
            # 随机决定是否使用探测提示
            use_detection_prompt = (torch.rand(1).item() > gt_prompt_ratio)
            
            outputs = self.model(images, coords, use_detection_prompt=use_detection_prompt)
            
            losses = self.criterion(outputs, masks, coords)
            loss = losses['total']
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累加分项损失
            total_loss_sum += loss.item()
            det_loss_sum += losses['detection']['total'].item()
            seg_loss_sum += losses['segmentation']['total'].item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'DetRatio': f'{det_prompt_ratio:.1%}'})
            
            # TensorBoard 记录
            if batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
        
        # 计算平均值
        num_batches = len(self.train_loader)
        avg_loss = total_loss_sum / num_batches
        avg_det_loss = det_loss_sum / num_batches
        avg_seg_loss = seg_loss_sum / num_batches
        
        return avg_loss, avg_det_loss, avg_seg_loss, det_prompt_ratio

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc='Validating', leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            coords = batch['coords'].to(self.device)
            
            # 验证时始终使用探测分支生成提示
            outputs = self.model(images, coords, use_detection_prompt=True)
            
            losses = self.criterion(outputs, masks, coords)
            total_loss += losses['total'].item()
            
            pred_masks = outputs['segmentation']
            # 确保尺寸一致
            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = F.interpolate(pred_masks, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            for i in range(pred_masks.shape[0]):
                metrics_calc.update(pred_masks[i:i+1], masks[i:i+1])
        
        metrics = metrics_calc.compute()
        avg_loss = total_loss / len(self.val_loader)
        
        self.writer.add_scalar('Val/IoU', metrics['IoU'], epoch)
        return metrics, avg_loss

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_iou': self.best_iou,
        }
        # 保存最新模型
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best.pth'))
            
        # 定期保存
        if epoch % self.config['logging']['save_interval'] == 0:
            torch.save(checkpoint, os.path.join(self.save_dir, f'epoch_{epoch}.pth'))

    def train(self):
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            epoch_start = time.time()
            
            # 1. 训练一个 Epoch (获取分项损失)
            train_loss, train_det_loss, train_seg_loss, det_ratio = self.train_epoch(epoch)
            
            # 2. 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 3. 计算时间
            epoch_time = time.time() - epoch_start
            
            # === 打印训练信息 (Line 1 & 2) ===
            self.logger.info("-" * 60)
            self.logger.info(
                f"Epoch [{epoch}/{self.config['training']['num_epochs']}] "
                f"Time: {epoch_time:.2f}s | LR: {current_lr:.6f} | Det Ratio: {det_ratio:.1%}"
            )
            
            self.logger.info(
                f"  [Train] Total: {train_loss:.4f} | Det: {train_det_loss:.4f} | Seg: {train_seg_loss:.4f}"
            )
            
            # 4. 验证与保存 (Line 3)
            if epoch % self.config['logging']['val_interval'] == 0:
                metrics, val_loss = self.validate(epoch)
                
                val_iou = metrics['IoU']
                val_pd = metrics['Pd']
                val_f1 = metrics['F1']
                val_fa = metrics['Fa']
                
                self.logger.info(
                    f"  [Val]   IoU: {val_iou:.2f}% | Pd: {val_pd:.2f}% | "
                    f"F1: {val_f1:.2f}% | Fa: {val_fa:.2f}"
                )
                
                # 保存最佳模型
                is_best = val_iou > self.best_iou
                if is_best:
                    self.best_iou = val_iou
                    self.logger.info(f"  >>> Best Model Saved (IoU: {self.best_iou:.2f}%)")
                
                self.save_checkpoint(epoch, is_best)
            else:
                self.save_checkpoint(epoch, is_best=False)
                
        self.logger.info(f"\nTraining completed! Best IoU: {self.best_iou:.2f}%")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    Trainer(args.config).train()

if __name__ == '__main__':
    main()