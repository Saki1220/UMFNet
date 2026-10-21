import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels * self.expansion,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, in_channels, block, layers):
        super().__init__()

        self.in_channels = in_channels

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.projinto1 = nn.Sequential(
            nn.Conv2d(in_channels, 64 * block.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(64 * block.expansion)
        )

        self.proj1to2 = nn.Sequential(
            nn.Conv2d(64 * block.expansion, 128 * block.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(128 * block.expansion)
        )

        self.proj2to3 = nn.Sequential(
            nn.Conv2d(128 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(256 * block.expansion)
        )

        self.proj3to4 = nn.Sequential(
            nn.Conv2d(256 * block.expansion, 512 * block.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(512 * block.expansion)
        )

        self.alpha1 = nn.Parameter(torch.tensor(0.1))
        self.alpha2 = nn.Parameter(torch.tensor(0.1))
        self.alpha3 = nn.Parameter(torch.tensor(0.1))
        self.alpha4 = nn.Parameter(torch.tensor(0.1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels,
                            stride=stride, downsample=downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        x1_input = self.layer1(x)
        x_proj1 = self.projinto1(
            F.interpolate(x, size=x1_input.shape[2:], mode='bilinear', align_corners=False)
        )
        x1 = x1_input + self.alpha1 * x_proj1

        x2_input = self.layer2(x1)
        x_proj2 = self.proj1to2(
            F.interpolate(x1, size=x2_input.shape[2:], mode='bilinear', align_corners=False)
        )
        x2 = x2_input + self.alpha2 * x_proj2

        x3_input = self.layer3(x2)
        x_proj3 = self.proj2to3(
            F.interpolate(x2, size=x3_input.shape[2:], mode='bilinear', align_corners=False)
        )
        x3 = x3_input + self.alpha3 * x_proj3

        x4_input = self.layer4(x3)
        x_proj4 = self.proj3to4(
            F.interpolate(x3, size=x4_input.shape[2:], mode='bilinear', align_corners=False)
        )
        x4 = x4_input + self.alpha4 * x_proj4

        return x1, x2, x3, x4

def resnet_custom(in_channels, layers = [3, 4, 6, 3]):
    return ResNet(in_channels, ConvBlock, layers)
