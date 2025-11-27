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

# --------------------------------------------------------------------------------
# 1. 设置日志记录 (参考 SAMIRNet)
# --------------------------------------------------------------------------------
def setup_logging(log_file):
    """设置日志，同时输出到控制台和文件"""
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    
    # 文件处理器
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    
    return logger

class Trainer:
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 1. 生成带时间戳的保存目录 (参考 SAMIRNet)
        run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        base_save_dir = self.config['logging']['save_dir']
        # 结构: experiments/2023-xx-xx_xx-xx-xx/
        self.save_dir = os.path.join(base_save_dir, run_timestamp)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 2. 初始化日志
        log_path = os.path.join(self.save_dir, 'train_log.log')
        self.logger = setup_logging(log_path)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Training Run ID: {run_timestamp}")
        self.logger.info(f"Save Directory: {self.save_dir}")
        self.logger.info(f"{'='*80}\n")
        
        # 打印超参数
        self.logger.info("Configuration:")
        for section, params in self.config.items():
            self.logger.info(f"[{section}]")
            if isinstance(params, dict):
                for k, v in params.items():
                    self.logger.info(f"  {k}: {v}")
        self.logger.info(f"{'-'*80}\n")

        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # 创建模型
        self.model = UnifiedModel(self.config).to(self.device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model Parameters: {total_params / 1e6:.4f}M (Trainable: {trainable_params / 1e6:.4f}M)\n")
        
        # 创建数据集
        self.train_dataset = IRSTDDataset(
            root_dir=self.config['dataset']['root_dir'],
            split=self.config['dataset']['train_split'],
            image_size=self.config['model']['image_size']
        )
        
        self.val_dataset = IRSTDDataset(
            root_dir=self.config['dataset']['root_dir'],
            split=self.config['dataset']['val_split'],
            image_size=self.config['model']['image_size']
        )
        
        # 创建数据加载器
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
        
        self.logger.info(f"Train samples: {len(self.train_dataset)}")
        self.logger.info(f"Val samples: {len(self.val_dataset)}\n")
        
        # 创建损失函数
        self.criterion = CombinedLoss(self.config).to(self.device)
        
        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=self.config['optimizer']['betas']
        )
        
        # 创建学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=self.config['scheduler']['eta_min']
        )
        
        # 创建Tensorboard Writer
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))
        
        self.current_epoch = 0
        self.best_iou = 0.0
    
    def get_prompt_strategy(self, epoch: int) -> float:
        """获取当前epoch使用GT prompt的比例"""
        schedule = self.config['training']['gt_prompt_ratio_schedule']
        start_epoch = schedule['start_epoch']
        end_epoch = int(self.config['training']['num_epochs'] * 0.1)
        
        if epoch < end_epoch:
            return 1.0
        else:
            progress = (epoch - end_epoch) / (self.config['training']['num_epochs'] - end_epoch)
            ratio = 1.0 - progress
            return max(0.0, ratio)
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        gt_prompt_ratio = self.get_prompt_strategy(epoch)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["num_epochs"]}', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            coords = batch['coords'].to(self.device)
            
            # 决定是否使用探测分支的输出
            use_detection_prompt = (torch.rand(1).item() > gt_prompt_ratio)
            
            # 前向传播
            outputs = self.model(images, coords, use_detection_prompt=use_detection_prompt)
            
            # 计算损失
            losses = self.criterion(outputs, masks, coords)
            loss = losses['total']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            det_loss_sum += losses['detection']['total'].item()
            seg_loss_sum += losses['segmentation']['total'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gt_ratio': f'{gt_prompt_ratio:.2f}'
            })
            
            # 记录详细日志到Tensorboard
            if batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/DetLoss', losses['detection']['total'].item(), global_step)
                self.writer.add_scalar('Train/SegLoss', losses['segmentation']['total'].item(), global_step)
                self.writer.add_scalar('Train/GTPromptRatio', gt_prompt_ratio, global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        avg_det_loss = det_loss_sum / len(self.train_loader)
        avg_seg_loss = seg_loss_sum / len(self.train_loader)
        
        return avg_loss, avg_det_loss, avg_seg_loss
    
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
            
            # 使用探测分支的输出
            outputs = self.model(images, coords, use_detection_prompt=True)
            
            # 计算损失
            losses = self.criterion(outputs, masks, coords)
            total_loss += losses['total'].item()
            
            # 计算指标
            pred_masks = outputs['segmentation']
            
            # 防御性编程：如果模型没在内部做插值，这里补救一下
            # (如果你已经修改了UnifiedModel的forward，这一步实际上是多余但无害的)
            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks, 
                    size=masks.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )

            for i in range(pred_masks.shape[0]):
                metrics_calc.update(pred_masks[i:i+1], masks[i:i+1])
        
        # 计算平均指标
        metrics = metrics_calc.compute()
        avg_loss = total_loss / len(self.val_loader)
        
        # 记录日志
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/IoU', metrics['IoU'], epoch)
        self.writer.add_scalar('Val/F1', metrics['F1'], epoch)
        self.writer.add_scalar('Val/Pd', metrics['Pd'], epoch)
        self.writer.add_scalar('Val/Fa', metrics['Fa'], epoch)
        
        return metrics, avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'config': self.config
        }
        
        # 1. 保存 Latest (最后一次)
        latest_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 2. 保存 Best (如果当前是最佳)
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"  >>> Best Model Saved (IoU: {self.best_iou:.2f}%)")
        
        # 3. 定期保存 (如 epoch_10.pth)
        if epoch % self.config['logging']['save_interval'] == 0:
            epoch_path = os.path.join(self.save_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        self.logger.info("Starting training...")
        
        start_time = time.time()
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # 训练
            train_loss, train_det_loss, train_seg_loss = self.train_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LR', current_lr, epoch)
            
            # 构建日志字符串 (仿 SAMIRNet 风格)
            epoch_time = time.time() - epoch_start
            log_str = f"Epoch [{epoch}/{self.config['training']['num_epochs']}] Time: {epoch_time:.2f}s | LR: {current_lr:.6f}\n"
            log_str += f"  [Train] Total: {train_loss:.4f} | Det: {train_det_loss:.4f} | Seg: {train_seg_loss:.4f}"
            
            # 验证
            if epoch % self.config['logging']['val_interval'] == 0:
                metrics, val_loss = self.validate(epoch)
                val_iou = metrics['IoU']
                
                # 保存最佳模型
                is_best = val_iou > self.best_iou
                if is_best:
                    self.best_iou = val_iou
                
                # 追加验证日志
                log_str += f"\n  [Val]   IoU: {metrics['IoU']:.2f}% | Pd: {metrics['Pd']:.2f}% | F1: {metrics['F1']:.2f}% | Fa: {metrics['Fa']:.2f}"
                
                # 打印完整日志
                self.logger.info(log_str)
                
                # 保存模型
                self.save_checkpoint(epoch, is_best)
            else:
                # 只打印训练日志
                self.logger.info(log_str)
                # 也可以选择每轮都保存 latest
                self.save_checkpoint(epoch, is_best=False)
                
            self.logger.info("-" * 60)
        
        total_time = time.time() - start_time
        self.logger.info(f"\nTraining completed in {total_time/3600:.2f} hours! Best IoU: {self.best_iou:.2f}%")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train SAM2 Detection-Segmentation Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()