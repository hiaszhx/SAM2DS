# models/sam2_model.py
import torch
import torch.nn as nn
import os

try:
    from .sam2_core.hieradet import Hiera, FpnNeck
except ImportError:
    from models.sam2_core.hieradet import Hiera, FpnNeck

class SAM2Wrapper(nn.Module):
    """
    SAM2模型封装 (双 Neck 版)：
    1. 共享 Backbone (Trunk)
    2. 独立 Detection Neck
    3. 独立 Segmentation Neck
    """
    
    def __init__(self, checkpoint_path: str, model_cfg: str, 
                 config_dir: str = None, 
                 freeze_trunk: bool = True, 
                 freeze_neck_det: bool = False,
                 freeze_neck_seg: bool = False):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        
        # 1. 实例化 Backbone (共享)
        self.trunk = Hiera(
            embed_dim=96,
            num_heads=1,
            stages=[1, 2, 7, 2],
            global_att_blocks=[5, 7, 9],
            window_pos_embed_bkg_spatial_size=[7, 7],
            window_spec=[8, 4, 14, 7]
        )
        
        # 2. 实例化两个独立的 Neck
        # Neck Config (保持一致)
        neck_config = dict(
            d_model=256,
            backbone_channel_list=[768, 384, 192, 96],
            fpn_top_down_levels=[2, 3] 
        )
        
        self.neck_det = FpnNeck(**neck_config) # 探测专用
        self.neck_seg = FpnNeck(**neck_config) # 分割专用
        
        # 3. 加载权重 (一份权重加载给两个Neck)
        self._load_weights()
        
        # 4. 冻结策略
        # 冻结 Backbone
        if freeze_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = False
            print("SAM2 Trunk frozen.")
        
        # 配置 Detection Neck
        if freeze_neck_det:
            for param in self.neck_det.parameters():
                param.requires_grad = False
            print("SAM2 Detection Neck frozen.")
        else:
            self.neck_det.train()
            for param in self.neck_det.parameters():
                param.requires_grad = True
            print("SAM2 Detection Neck unfrozen.")

        # 配置 Segmentation Neck
        if freeze_neck_seg:
            for param in self.neck_seg.parameters():
                param.requires_grad = False
            print("SAM2 Segmentation Neck frozen.")
        else:
            self.neck_seg.train()
            for param in self.neck_seg.parameters():
                param.requires_grad = True
            print("SAM2 Segmentation Neck unfrozen.")
            
        self.encoder_embed_dim = 256

    def _load_weights(self):
        if not os.path.exists(self.checkpoint_path):
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}, using random init.")
            return

        print(f"Loading SAM2 checkpoint from {self.checkpoint_path}...")
        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        
        if "model" in state_dict:
            state_dict = state_dict["model"]
            
        trunk_dict = {}
        neck_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith("image_encoder.trunk."):
                new_k = k.replace("image_encoder.trunk.", "")
                trunk_dict[new_k] = v
            elif k.startswith("image_encoder.neck."):
                new_k = k.replace("image_encoder.neck.", "")
                neck_dict[new_k] = v
                
        if trunk_dict:
            msg = self.trunk.load_state_dict(trunk_dict, strict=False)
            print(f"Trunk loaded: {msg}")
            
        if neck_dict:
            # === 关键修改：将同一份 Neck 权重加载到两个模块中 ===
            msg1 = self.neck_det.load_state_dict(neck_dict, strict=False)
            msg2 = self.neck_seg.load_state_dict(neck_dict, strict=False)
            print(f"Dual Necks loaded initialized with same pretrained weights.")
            print(f"  - Det Neck: {msg1}")
            print(f"  - Seg Neck: {msg2}")
    
    def forward(self, x: torch.Tensor):
        # 1. 共享 Backbone 提取特征
        backbone_out = self.trunk(x)
        
        # 2. 独立 Neck 处理
        features_det = self.neck_det(backbone_out) # 用于探测的多尺度特征
        features_seg = self.neck_seg(backbone_out) # 用于分割的多尺度特征
        
        # 返回两个列表
        return features_det, features_seg

    def get_encoder_output_dim(self) -> int:
        return self.encoder_embed_dim