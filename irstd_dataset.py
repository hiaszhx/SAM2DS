import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class IRSTDDataset(Dataset):
    """红外小目标检测数据集"""
    
    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 256):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # 数据路径
        self.image_dir = os.path.join(root_dir, 'images', split)
        self.mask_dir = os.path.join(root_dir, 'masks', split)
        
        # 获取文件列表
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.bmp'))])
        
        # 数据增强
        if split == 'train':
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # 加载掩码
        mask_name = img_name.replace('.png', '.png').replace('.jpg', '.png').replace('.bmp', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32)
        
        # 数据增强
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)
        
        # 计算目标中心点坐标 (归一化)
        coords = self._get_target_center(mask)
        
        return {
            'image': image,
            'mask': mask,
            'coords': coords,
            'name': img_name
        }
    
    def _get_target_center(self, mask: torch.Tensor) -> torch.Tensor:
        """获取目标中心点坐标"""
        mask_np = mask.squeeze().numpy()
        
        if mask_np.sum() == 0:
            # 如果没有目标，返回中心点
            return torch.tensor([0.5, 0.5], dtype=torch.float32)
        
        # 计算质心
        y_indices, x_indices = np.where(mask_np > 0.5)
        center_x = x_indices.mean() / mask_np.shape[1]
        center_y = y_indices.mean() / mask_np.shape[0]
        
        return torch.tensor([center_x, center_y], dtype=torch.float32)
