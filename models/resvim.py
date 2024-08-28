import torch

import torchvision.models as models

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
from einops.layers.torch import Reduce


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def output_head(dim: int, num_classes: int):
    return nn.Sequential(
        Reduce("b s d -> b d", "mean"),
        nn.LayerNorm(dim),
        nn.Linear(dim, num_classes),
    )


class VisionEncoderMambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim
        self.dt_rank = dt_rank
        self.dim_inner = dim_inner
        self.d_state = d_state

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        self.silu = nn.SiLU()
        self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

        self.proj = nn.Linear(dim, dim)

        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape

        skip = x

        x = self.norm(x)

        z1 = self.proj(x)
        x = self.proj(x)

        x1 = self.process_direction(
            x,
            self.forward_conv1d,
            self.ssm,
        )

        x2 = self.process_direction(
            x,
            self.backward_conv1d,
            self.ssm,
        )

        z = self.silu(z1)

        x1 *= z
        x2 *= z
        
        return x1 + x2 + skip

    def process_direction(
        self,
        x: Tensor,
        conv1d: nn.Conv1d,
        ssm: SSM,
    ):
        x = rearrange(x, "b s d -> b d s")
        x = self.softplus(conv1d(x))
        x = rearrange(x, "b d s -> b s d")
        x = ssm(x)
        return x
    
class VimRes(nn.Module):
    def __init__(
        self,
        dim: int,
        dt_rank: int = 32,
        dim_inner: int = None,
        d_state: int = None,
        num_classes: int = None,
        image_size: int = 28,
        patch_size: int = 4,
        channels: int = 3,
        dropout: float = 0.1,
        depth: int = 12,
        resnet_model='resnet18'
        *args,
        **kwargs,
    ):
        resnet = getattr(models, resnet_type)(pretrained=True)
        self.resnet_stream = nn.Sequential(*list(resnet.children())[:-2])