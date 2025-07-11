# from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
# from .resnetv2 import ResNet50
# from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
# from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
# from .mobilenetv2 import mobile_half
# from .ShuffleNetv1 import ShuffleV1
# from .ShuffleNetv2 import ShuffleV2

# Import các model classes
from .vit import ViT, create_vit_teacher  # Mô hình teacher
from .vit_small import ViT as ViT_small, create_vit_student  # Mô hình student
from .vit_timm import ViTTimm, create_vit_timm_teacher, SUPPORTED_MODELS, get_model_info  # ViT pretrained từ timm

# Factory functions cho teacher models
def get_vit_teacher(num_classes=14, pretrained=False, patch_size=16):
    """
    Trả về mô hình ViT làm teacher với cấu trúc được tối ưu cho ChestX-ray14
    
    Args:
        num_classes (int): Số lớp đầu ra (14 cho ChestX-ray14)
        pretrained (bool): Có sử dụng pretrained weights hay không
        patch_size (int): Kích thước patch cho ViT
    """
    return create_vit_teacher(
        num_classes=num_classes,
        pretrained=pretrained,
        patch_size=patch_size
    )

def get_vit_timm_teacher(num_classes=14, pretrained=True, model_name='vit_base_patch16_224'):
    """
    Trả về mô hình ViT pretrained từ timm làm teacher
    
    Args:
        num_classes (int): Số lớp đầu ra (14 cho ChestX-ray14)
        pretrained (bool): Có sử dụng pretrained weights hay không
        model_name (str): Tên model từ timm
    """
    return create_vit_timm_teacher(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )

# Factory function cho student model
def get_vit_student(num_classes=14, pretrained=False):
    """
    Trả về mô hình ViT_small làm student cho ChestX-ray14
    """
    return create_vit_student(
        num_classes=num_classes,
        pretrained=pretrained
    )

# Model dictionary
model_dict = {
    'vit_teacher': get_vit_teacher,
    'vit_timm_teacher': get_vit_timm_teacher,  # Thêm timm pretrained teacher
    'vit_student': get_vit_student,
    'vit_small': get_vit_student  # Thêm vit_small để train baseline
}
