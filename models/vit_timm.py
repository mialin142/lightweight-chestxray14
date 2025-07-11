# ViT pretrained từ timm cho ChestX-ray14
# Hỗ trợ các model: vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224
# và DeiT: deit_base_patch16_224, deit_large_patch16_224

import torch
import torch.nn as nn
from typing import List, Tuple, Optional

# Import timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Install with: pip install timm")


class ViTTimm(nn.Module):
    """
    ViT pretrained từ timm với các tính năng bổ sung cho knowledge distillation
    """
    
    def __init__(self, 
                 model_name: str = 'vit_base_patch16_224',
                 num_classes: int = 14,
                 pretrained: bool = True,
                 drop_rate: float = 0.1,
                 drop_path_rate: float = 0.1,
                 img_size: int = 224,
                 patch_size: int = 16):
        """
        Args:
            model_name: Tên model từ timm
            num_classes: Số lớp đầu ra (14 cho ChestX-ray14)
            pretrained: Có sử dụng pretrained weights hay không
            drop_rate: Dropout rate cho classification head
            drop_path_rate: Drop path rate cho transformer blocks
            img_size: Kích thước ảnh đầu vào
            patch_size: Kích thước patch (được xác định bởi model_name)
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required for ViTTimm. Install with: pip install timm")
        
        # Tạo model từ timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            img_size=img_size
        )
        
        # Lưu thông tin model
        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Lấy số layers để xác định intermediate features
        if hasattr(self.model, 'blocks'):
            self.num_layers = len(self.model.blocks)
        else:
            self.num_layers = 12  # Default cho ViT base
        
        print(f"Created ViTTimm model: {model_name}")
        print(f"Number of layers: {self.num_layers}")
        print(f"Input size: {img_size}x{img_size}")
        print(f"Patch size: {patch_size}x{patch_size}")
    
    def forward(self, x: torch.Tensor, is_feat: bool = False) -> Tuple[List[torch.Tensor], torch.Tensor] | torch.Tensor:
        """
        Forward pass với tùy chọn trả về intermediate features
        
        Args:
            x: Input tensor (B, C, H, W)
            is_feat: Có trả về intermediate features hay không
            
        Returns:
            Nếu is_feat=True: (features, logits)
            Nếu is_feat=False: logits
        """
        if is_feat:
            return self.forward_with_features(x)
        else:
            return self.model(x)
    
    def forward_with_features(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass trả về intermediate features cho knowledge distillation
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            features: List các intermediate features
            logits: Output logits
        """
        features = []
        
        # Patch embedding
        if hasattr(self.model, 'patch_embed'):
            x = self.model.patch_embed(x)
            features.append(x)  # Patch embeddings
        
        # Position embedding và dropout
        if hasattr(self.model, 'pos_drop'):
            x = self.model.pos_drop(x)
        
        # Transformer blocks
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks):
                x = block(x)
                # Lấy features từ một số layers quan trọng
                if i in [self.num_layers // 4, self.num_layers // 2, 3 * self.num_layers // 4]:
                    features.append(x)
        
        # Final norm và head
        if hasattr(self.model, 'norm'):
            x = self.model.norm(x)
        
        # Global average pooling hoặc CLS token
        if hasattr(self.model, 'head'):
            if hasattr(self.model, 'global_pool'):
                if self.model.global_pool == 'avg':
                    x = x.mean(dim=1)  # Global average pooling
                else:
                    x = x[:, 0]  # CLS token
            else:
                x = x[:, 0]  # Default: CLS token
            
            features.append(x)  # Pre-classifier features
            logits = self.model.head(x)
        else:
            logits = x
        
        return features, logits
    
    def get_feature_dim(self) -> int:
        """Trả về dimension của features"""
        if hasattr(self.model, 'num_features'):
            return self.model.num_features
        elif hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim
        else:
            return 768  # Default cho ViT base
    
    def get_patch_embed_dim(self) -> int:
        """Trả về dimension của patch embeddings"""
        if hasattr(self.model, 'patch_embed'):
            if hasattr(self.model.patch_embed, 'proj'):
                return self.model.patch_embed.proj.out_channels
        return self.get_feature_dim()


def create_vit_timm_teacher(model_name: str = 'vit_base_patch16_224',
                           num_classes: int = 14,
                           pretrained: bool = True,
                           **kwargs) -> ViTTimm:
    """
    Factory function để tạo ViT teacher từ timm
    
    Args:
        model_name: Tên model từ timm
        num_classes: Số lớp đầu ra
        pretrained: Có sử dụng pretrained weights hay không
        **kwargs: Các tham số khác
        
    Returns:
        ViTTimm model
    """
    return ViTTimm(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs
    )


# Danh sách các model timm được hỗ trợ
SUPPORTED_MODELS = [
    'vit_base_patch16_224',
    'vit_large_patch16_224', 
    'vit_huge_patch14_224',
    'deit_base_patch16_224',
    'deit_large_patch16_224',
    'deit_base_distilled_patch16_224',
    'deit_large_distilled_patch16_224'
]


def is_supported_model(model_name: str) -> bool:
    """Kiểm tra xem model có được hỗ trợ hay không"""
    return model_name in SUPPORTED_MODELS


def get_model_info(model_name: str) -> dict:
    """Trả về thông tin về model"""
    model_info = {
        'vit_base_patch16_224': {
            'params': '86M',
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'heads': 12
        },
        'vit_large_patch16_224': {
            'params': '304M', 
            'patch_size': 16,
            'embed_dim': 1024,
            'depth': 24,
            'heads': 16
        },
        'vit_huge_patch14_224': {
            'params': '632M',
            'patch_size': 14, 
            'embed_dim': 1280,
            'depth': 32,
            'heads': 16
        },
        'deit_base_patch16_224': {
            'params': '86M',
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'heads': 12
        },
        'deit_large_patch16_224': {
            'params': '304M',
            'patch_size': 16, 
            'embed_dim': 1024,
            'depth': 24,
            'heads': 16
        }
    }
    
    return model_info.get(model_name, {}) 