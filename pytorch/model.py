import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.expansion = 4
        self.conv_1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False, 
            **kwargs
        )
        self.bt1 = nn.BatchNorm2d(intermediate_channels)
        self.relu1 = nn.ReLU()

        self.conv_2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            **kwargs
        )
        self.bt2 = nn.BatchNorm2d(intermediate_channels)
        self.relu2 = nn.ReLU()

        self.conv_3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels*self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            **kwargs
        )
        self.bt3 = nn.BatchNorm2d(intermediate_channels*self.expansion)
        self.relu3 = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv_1(x)
        x = self.bt1(x)
        x = self.relu1(x)

        x = self.conv_2(x)
        x = self.bt2(x)
        x = self.relu2(x)

        x = self.conv_3(x)
        x = self.bt3(x)

        # Pass the input through a downsampling layer if it is not None
        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu3(identity)
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_layers, n_channels, num_classes):
        super().__init__()

        # Initial convolution layers
        self.in_channels = 64
        self.conv_1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bt1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Redisual blocks

        self.conv_2 = self._make_block(block, num_layers[0], intermediate_channels=64, stride=1)
        self.conv_3 = self._make_block(block, num_layers[1], intermediate_channels=128, stride=2)
        self.conv_4 = self._make_block(block, num_layers[2], intermediate_channels=256, stride=2)
        self.conv_5 = self._make_block(block, num_layers[3], intermediate_channels=512, stride=2)

        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bt1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        # Fully connected layers
        x = self.average_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_block(self, residual_block, num_blocks, intermediate_channels, stride):
        downsample = None
        layers = []

        # Pass the input through a downsampling layer to match the dimensions for the addition to be possible
        if stride != 1 or self.in_channels != intermediate_channels*4:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=intermediate_channels*4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels*4)
            )

        layers.append(
            residual_block(self.in_channels, intermediate_channels, stride, downsample)
        )

        self.in_channels = intermediate_channels*4

        for _ in range(num_blocks-1):
            layers.append(
                residual_block(self.in_channels, intermediate_channels)
            )

        return nn.Sequential(*layers)
            


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = ResNet(ResidualBlock, [3, 4, 6, 3], 3, 1000).to(device)
    # x = torch.rand((1, 3, 224, 224), device=device)
    # y = net(x)
    # print(y.shape)

    log_tensorboard = False

    if log_tensorboard:
        writer = SummaryWriter('logs/resnet50')

        writer.add_graph(net.to("cuda"), torch.rand(1, 3, 224, 224, device="cuda"))
        writer.close()