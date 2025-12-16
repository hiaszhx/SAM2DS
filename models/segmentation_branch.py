# models/segmentation_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .transformer_utils import TwoWayTransformer
from .sam_mask_decoder import MaskDecoder

class PositionEmbeddingSine(nn.Module):
    """
    标准的正弦位置编码，用于 SAM 图像特征和 Prompt 坐标
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor):
        # x: [B, C, H, W] 或 [B, C] (如果是坐标)
        # 这里仅处理 [B, C, H, W] 生成 grid PE，或者 [B, N, 2] 生成 point PE
        pass
    
    def forward_with_coords(self, coords_input, image_size):
        """
        Args:
            coords_input: [B, N, 2] 归一化坐标 [0, 1]
            image_size: (H, W)
        Returns:
            pe: [B, N, C]
        """
        coords = coords_input.clone()
        # 反归一化为 pixel 坐标，这里简单处理，实际上SAM是直接用归一化的做sin/cos
        # 但标准做法是保持归一化并乘上 scale
        coords = coords * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=coords.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = coords[:, :, 0, None] / dim_t
        pos_y = coords[:, :, 1, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2)
        return pos

    def forward_image(self, h, w, device):
        """生成图像的 Grid PE"""
        y_embed = torch.arange(1, h + 1, dtype=torch.float32, device=device)
        x_embed = torch.arange(1, w + 1, dtype=torch.float32, device=device)
        
        if self.normalize:
            y_embed = y_embed / (h + 1e-6) * self.scale
            x_embed = x_embed / (w + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        
        pos = torch.cat((pos_y.unsqueeze(1).repeat(1, w, 1), pos_x.unsqueeze(0).repeat(h, 1, 1)), dim=2)
        return pos.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]

class SegmentationBranch(nn.Module):
    """
    基于 Two-Way Transformer 的分割分支
    """
    def __init__(self, in_channels: int, adapter_dim: int = 256):
        super().__init__()
        
        self.transformer_dim = adapter_dim
        
        # 1. 适配层：确保通道数匹配 Transformer 维度
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, adapter_dim, 1),
            nn.BatchNorm2d(adapter_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. 位置编码器 (一半的维度给x，一半给y)
        self.pe_layer = PositionEmbeddingSine(num_pos_feats=adapter_dim // 2, normalize=True)
        
        # 3. Prompt Embeddings
        # 0: background point (not used usually for single point positive), 1: foreground point
        self.point_embeddings = nn.Embedding(2, adapter_dim) 
        self.not_a_point_embed = nn.Embedding(1, adapter_dim) # 用于填充 padding (如果需要)

        # 4. Transformer
        transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=adapter_dim,
            num_heads=8,
            mlp_dim=2048,
        )

        # 5. Mask Decoder
        self.mask_decoder = MaskDecoder(
            transformer_dim=adapter_dim,
            transformer=transformer,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        
    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool):
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel (optional, matches SAM logic)
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -1 * torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
            
        point_embedding = self.pe_layer.forward_with_coords(points, None) # [B, N, C]
        
        # Add label embedding
        # labels: 1 for foreground, 0 for background.
        # We assume input 'labels' are 1 (positive).
        point_embedding[labels == 0] += self.point_embeddings(torch.tensor(0, device=points.device))
        point_embedding[labels == 1] += self.point_embeddings(torch.tensor(1, device=points.device))
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        
        return point_embedding

    def forward(self, x: torch.Tensor, prompt_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 融合后的特征 [B, C, H, W]
            prompt_coords: 归一化坐标 [B, 2] (range 0-1)
        Returns:
            mask: [B, 1, H_out, W_out]
        """
        # 1. Adapter & Features
        src = self.adapter(x)
        B, C, H, W = src.shape
        
        # 2. Image Positional Encoding
        image_pe = self.pe_layer.forward_image(H, W, x.device) # [1, C, H, W]
        image_pe = image_pe.repeat(B, 1, 1, 1)
        
        # 3. Prompt Embedding
        # prompt_coords is [B, 2], treat as [B, 1, 2]
        # Label is always 1 (Foreground)
        coords = prompt_coords.unsqueeze(1) # [B, 1, 2]
        labels = torch.ones(B, 1, dtype=torch.long, device=x.device)
        
        sparse_prompt_embeddings = self._embed_points(coords, labels, pad=False)
        
        # Dense embeddings (None/Zero for now as we don't have mask prompt)
        dense_prompt_embeddings = torch.zeros_like(src)
        
        # 4. Decoder Forward
        low_res_masks, iou_pred = self.mask_decoder(
            image_embeddings=src,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False, # 训练时通常选单mask，或者根据策略选
        )
        
        # low_res_masks is usually 1/4 resolution of input 'x' (due to transformer architecture)
        # Check output shape. MaskDecoder usually upsamples 4x from token space.
        
        return low_res_masks