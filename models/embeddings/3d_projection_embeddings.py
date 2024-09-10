import torch.nn as nn

class ProjectionModule(nn.Module):
    def __init__(self):
        super(ProjectionModule, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(28, 1, 1), stride=(1, 1, 1))

    def forward(self, x):
        x = self.conv3d(x)
        x = x.squeeze(2)
        return x