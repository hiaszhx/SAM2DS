# models/sam2_core/hieradet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .utils import MLP, DropPath, PatchEmbed, window_partition, window_unpartition

def do_pool(x, pool, norm=None):
    if pool is None: return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    if norm: x = norm(x)
    return x

class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_pool=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
        )
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        x = self.proj(x)
        return x

class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0, drop_path=0.0, norm_layer="LayerNorm", q_stride=None, act_layer=nn.GELU, window_size=0):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Hiera(nn.Module):
    def __init__(self, embed_dim=96, num_heads=1, drop_path_rate=0.0, q_pool=3, q_stride=(2, 2), stages=(2, 3, 16, 3), dim_mul=2.0, head_mul=2.0, window_pos_embed_bkg_spatial_size=(14, 14), window_spec=(8, 4, 14, 7), global_att_blocks=(12, 16, 20), return_interm_layers=True):
        super().__init__()
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            block = MultiScaleBlock(dim=embed_dim, dim_out=dim_out, num_heads=num_heads, drop_path=dpr[i], q_stride=self.q_stride if i in self.q_pool_blocks else None, window_size=window_size)
            embed_dim = dim_out
            self.blocks.append(block)
        self.channel_list = [self.blocks[i].dim_out for i in self.stage_ends[::-1]] if return_interm_layers else [self.blocks[-1].dim_out]

    def _get_pos_embed(self, hw):
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        return outputs

class FpnNeck(nn.Module):
    def __init__(self, d_model, backbone_channel_list, kernel_size=1, stride=1, padding=0, fpn_interp_model="bilinear", fuse_type="sum", fpn_top_down_levels=None):
        super().__init__()
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            # 关键修改：使用 Sequential 包装并命名为 "conv"，以匹配预训练权重的键名
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(dim, d_model, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            self.convs.append(current)
            
        self.fpn_interp_model = fpn_interp_model
        self.fuse_type = fuse_type
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs):
        out = [None] * len(self.convs)
        prev_features = None
        # FPN 是从深层(低分辨率)向浅层(高分辨率)处理
        # backbone_channel_list = [768, 384, 192, 96]
        # self.convs[0] 对应 768 (深层)
        # xs 必须是 [768, 384, 192, 96] 顺序 (Deep -> Shallow)
        # 或者 xs 是 [96, 192, 384, 768] (Shallow -> Deep)
        
        # 这里的循环逻辑: for i in range(3, -1, -1) -> 3, 2, 1, 0
        # 如果 i=3, conv_idx = n-i = 0. self.convs[0] 是 768通道.
        # 所以 x = xs[3] 必须是 768通道.
        # 这意味着 xs 的最后一个元素 xs[-1] 必须是最深层特征.
        
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(prev_features.to(dtype=torch.float32), scale_factor=2.0, mode=self.fpn_interp_model, align_corners=(None if self.fpn_interp_model == "nearest" else False), antialias=False)
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg": prev_features /= 2
            else:
                prev_features = lateral_features
            out[i] = prev_features
        return out