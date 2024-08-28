from torchvision.models import densenet121
import torch

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        self.densenet = densenet121(pretrained=False)
        
        self.densenet.features.conv0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3)        
        self.densenet.classifier = torch.nn.Linear(self.densenet.classifier.in_features, num_classes)

    def forward(self, x):
        return self.densenet(x)