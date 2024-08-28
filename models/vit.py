from torchvision.models import VisionTransformer
import torch

class ViT(VisionTransformer):
    def __init__(self, image_size=28, patch_size=4, num_classes=14):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=num_classes,
            dropout=0.0,
            attention_dropout=0.0,
        )
        
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.num_patches + 1, self.hidden_dim))
        self.patch_embed = torch.nn.Conv2d(3, self.hidden_dim, kernel_size=patch_size, stride=patch_size)