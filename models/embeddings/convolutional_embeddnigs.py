import torch

class ConvolutionalPatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=3, embed_dim=768):
        super(ConvolutionalPatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.conv = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        embeddings = self.conv(x)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        return embeddings