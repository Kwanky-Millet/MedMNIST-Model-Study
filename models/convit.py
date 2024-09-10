import torch.nn as nn
import timm

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=16, in_chans=3, embed_dim=432):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.norm = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x) 
        return x
    

class ConViT:
    def __init__(self, num_classes):
        self.model = timm.models.convit_small(num_classes=n_classes)
        self.model.patch_embed = PatchEmbed()
