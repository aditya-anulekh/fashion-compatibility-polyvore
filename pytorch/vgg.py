from typing import ForwardRef
import torch
from torch._C import device
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vgg16


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv):
        super(ConvBlock, self).__init__()
        self.out_channels = out_channels
        self.num_conv = num_conv
        self.in_channels = in_channels
        
        self.block = nn.Sequential(*self._make_layers())

    def forward(self, x):
        x = self.block(x)
        return x

    def _make_layers(self):
        layers = []
        for _ in range(self.num_conv):
            layers += [
                nn.Conv2d(in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU()]
            self.in_channels = self.out_channels

        return layers


class VGGNet(nn.Module):
    def __init__(self, architecture, num_classes):
        super(VGGNet, self).__init__()
        self.architecture = architecture
        self.model = self._make_layers()
        self.avg = nn.AdaptiveAvgPool2d((2,2))

        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = self.avg(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layers(self):
        layers = []
        in_channels = 3
        for (num_conv, out_channels) in self.architecture:
            layers += [
                ConvBlock(in_channels, out_channels, num_conv),
            ]
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        return nn.Sequential(*layers)


class DualVggNet(nn.Module):
    def __init__(self):
        super(DualVggNet, self).__init__()
        vgg_19 = [(2, 64), (2, 128), (3, 256), (3, 256), (3, 256)]
        self.model1 = vgg16(pretrained=False)
        self.model1.fc = nn.Identity()
        self.fc = nn.Sequential(
                            nn.Linear(2000,256),
                            nn.ReLU(),
                            # nn.Linear(512,256),
                            # nn.Dropout(0.3),
                            # nn.ReLU(),
                            nn.Linear(256,64),
                            nn.ReLU(),
                            nn.Linear(64,2),
                            # nn.Sigmoid()
                            )

    def forward(self, x, y):
        x = self.model1(x)
        y = self.model1(y)
        output = self.fc(torch.cat([x,y], 1))
        return output
        





if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_d = DualVggNet()
    model_d.to("cuda")
    x = torch.rand((2,3,224,224), device="cuda")
    y = torch.rand((2,3,224,224), device="cuda")



    log_tensorboard = True

    if log_tensorboard:
        writer = SummaryWriter('logs/vgg')

        writer.add_graph(model_d.to(device=device), torch.rand(1, 3, 224, 224, device=device))
        writer.close()
        