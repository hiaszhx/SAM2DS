import torch
import torch.nn as nn
from .sam2_model import SAM2Wrapper, SAM2EncoderWithResize
from .detection_branch import DetectionBranch
from .segmentation_branch import SegmentationBranch

class UnifiedModel(nn.Module):
    """统一的探测-分割模型"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        # SAM2编码器
        sam2_wrapper = SAM2Wrapper(
            checkpoint_path=config['model']['sam2_checkpoint'],
            model_cfg=config['model']['sam2_model_cfg'],
            config_dir=config['model'].get('sam2_config_dir', None),
            freeze_encoder=config['model']['freeze_image_encoder']
        )
        
        # 带尺寸调整的编码器
        self.encoder = SAM2EncoderWithResize(
            sam2_wrapper=sam2_wrapper,
            target_size=config['model']['input_size']
        )
        
        encoder_dim = self.encoder.get_encoder_output_dim()
        print(f"Encoder output dimension: {encoder_dim}")
        
        # 探测分支
        self.detection_branch = DetectionBranch(
            in_channels=encoder_dim,
            hidden_dim=config['model']['detection_hidden_dim'],
            adapter_dim=config['model']['adapter_dim']
        )
        
        # 分割分支
        self.segmentation_branch = SegmentationBranch(
            in_channels=encoder_dim,
            adapter_dim=config['model']['adapter_dim']
        )
        
        print("Unified model initialized successfully!")
    
    def forward(self, images: torch.Tensor, prompt_coords: torch.Tensor = None, 
                use_detection_prompt: bool = False) -> dict:
        """
        Args:
            images: 输入图像 [B, 3, H, W]
            prompt_coords: GT坐标 [B, 2] (可选)
            use_detection_prompt: 是否使用探测分支的输出作为prompt
        Returns:
            dict: {
                'detection': 探测结果,
                'segmentation': 分割结果,
                'prompt_coords': 使用的prompt坐标
            }
        """
        # 编码
        encoder_features = self.encoder(images)
        
        # 探测分支
        detection_output = self.detection_branch(encoder_features)
        
        # 确定使用的prompt
        if use_detection_prompt:
            # 使用探测分支输出
            prompt_coords_used = detection_output['coords'].detach()
        else:
            # 使用GT
            assert prompt_coords is not None, "GT prompt required when not using detection"
            prompt_coords_used = prompt_coords
        
        # 分割分支
        segmentation_output = self.segmentation_branch(encoder_features, prompt_coords_used)
        
        return {
            'detection': detection_output,
            'segmentation': segmentation_output,
            'prompt_coords': prompt_coords_used
        }
