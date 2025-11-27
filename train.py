import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from models.unified_model import UnifiedModel
from irstd_dataset import IRSTDDataset
from utils.losses import CombinedLoss
from utils.metrics import MetricsCalculator

class Trainer:
    def __init__(self, config_path: str):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建模型
        self.model = UnifiedModel(self.config).to(self.device)
        
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
        
        # 创建日志
        self.save_dir = self.config['logging']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'logs'))
        
        self.current_epoch = 0
        self.best_iou = 0.0
    
    def get_prompt_strategy(self, epoch: int) -> float:
        """
        获取当前epoch使用GT prompt的比例
        前10% epochs全部使用GT，之后逐渐降低
        """
        schedule = self.config['training']['gt_prompt_ratio_schedule']
        start_epoch = schedule['start_epoch']
        end_epoch = int(self.config['training']['num_epochs'] * 0.1)  # 前10% epochs
        
        if epoch < end_epoch:
            return 1.0  # 全部使用GT
        else:
            # 线性衰减
            progress = (epoch - end_epoch) / (self.config['training']['num_epochs'] - end_epoch)
            ratio = 1.0 - progress
            return max(0.0, ratio)
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        gt_prompt_ratio = self.get_prompt_strategy(epoch)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["num_epochs"]}')
        
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
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gt_ratio': f'{gt_prompt_ratio:.2f}'
            })
            
            # 记录日志
            if batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/DetLoss', losses['detection']['total'].item(), global_step)
                self.writer.add_scalar('Train/SegLoss', losses['segmentation']['total'].item(), global_step)
                self.writer.add_scalar('Train/GTPromptRatio', gt_prompt_ratio, global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc='Validating')
        
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
        
        print(f"\nValidation - Loss: {avg_loss:.4f}, IoU: {metrics['IoU']:.2f}%, "
              f"F1: {metrics['F1']:.2f}%, Pd: {metrics['Pd']:.2f}%, Fa: {metrics['Fa']:.2f}")
        
        return metrics['IoU'], avg_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with IoU: {self.best_iou:.2f}%")
        
        # 定期保存
        if epoch % self.config['logging']['save_interval'] == 0:
            epoch_path = os.path.join(self.save_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        print("Starting training...")
        print(f"Total epochs: {self.config['training']['num_epochs']}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            
            # 验证
            if epoch % self.config['logging']['val_interval'] == 0:
                val_iou, val_loss = self.validate(epoch)
                
                # 保存最佳模型
                is_best = val_iou > self.best_iou
                if is_best:
                    self.best_iou = val_iou
                
                self.save_checkpoint(epoch, is_best)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LR', current_lr, epoch)
        
        print(f"\nTraining completed! Best IoU: {self.best_iou:.2f}%")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train SAM2 Detection-Segmentation Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()
