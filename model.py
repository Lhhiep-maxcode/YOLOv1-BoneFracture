import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class Yolov1(nn.Module):
    def __init__(self, in_channels=1, split_size=10, num_boxes=2, num_classes=1):
        super(Yolov1, self).__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        original_conv1 = net.conv1
        net.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        modules = list(net.children())
        self.backbone = nn.Sequential(*modules[:-2])
        self.head = nn.Sequential(
            CNNBlock(in_channels=2048, out_channels=1024, kernel_size=1, padding=0, stride=1),  # (20, 20, 1024)
            CNNBlock(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1),  # (20, 20, 512)
            CNNBlock(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=2),   # (10, 10, 512)
            CNNBlock(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=1),    # (10, 10, 256)
            nn.Conv2d(in_channels=256, out_channels=11, kernel_size=1, padding=0, stride=1),    # (10, 10, 11)
        )

    def forward(self, x):
        x = self.head(self.backbone(x))
        return x.permute(0, 2, 3, 1).contiguous()
    
    
    