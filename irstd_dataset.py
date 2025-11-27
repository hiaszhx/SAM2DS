import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class IRSTDDataset(Dataset):
    """
    红外小目标检测数据集 (适配 BasicIRSTD/SAMIRNet 格式)
    结构:
        root_dir/
            images/ (所有图片)
            masks/  (所有掩码)
            img_idx/
                train_NUDT-SIRST.txt
                test_NUDT-SIRST.txt
    """
    
    def __init__(self, root_dir: str, split: str = 'train', image_size: int = 256):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        
        # 1. 推断数据集名称 (假设 root_dir 的最后一级目录名就是数据集名，例如 NUDT-SIRST)
        self.dataset_name = os.path.basename(os.path.normpath(root_dir))
        
        # 2. 设置路径
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # 3. 读取 img_idx 下的 txt 文件列表
        # txt文件名格式通常为: {split}_{dataset_name}.txt
        txt_name = f"{split}_{self.dataset_name}.txt"
        list_file = os.path.join(root_dir, 'img_idx', txt_name)
        
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"Dataset index file not found: {list_file}")
            
        with open(list_file, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
            
        print(f"Loaded {len(self.image_files)} images for {split} from {self.dataset_name}")
        
        # 4. 数据增强 (保持原 SAM2DS 逻辑不变)
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
        # 获取文件名 (txt中通常不带后缀，或者带后缀，需要适配)
        img_name = self.image_files[idx]
        
        # 1. 查找图片和掩码的真实路径
        img_path = self._find_image_path(self.image_dir, img_name)
        mask_path = self._find_image_path(self.mask_dir, img_name)
        
        # 2. 加载图像
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # 3. 加载掩码
        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 127).astype(np.float32)
        
        # 4. 数据增强
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask'].unsqueeze(0)
        
        # 5. 计算目标中心点坐标 (归一化)
        coords = self._get_target_center(mask)
        
        return {
            'image': image,
            'mask': mask,
            'coords': coords,
            'name': img_name
        }
    
    def _find_image_path(self, directory, img_name):
        """辅助函数：查找不同后缀的图片文件"""
        # 如果 txt 里已经包含了后缀 (例如 00001.png)
        full_path = os.path.join(directory, img_name)
        if os.path.exists(full_path):
            return full_path
            
        # 如果 txt 里只有文件名 (例如 00001)，尝试常见后缀
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        for ext in extensions:
            full_path = os.path.join(directory, img_name + ext)
            if os.path.exists(full_path):
                return full_path
                
        raise FileNotFoundError(f"Cannot find image file for '{img_name}' in '{directory}'")

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