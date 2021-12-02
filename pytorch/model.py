import torch.nn as nn
import torch
from torchvision.models import resnet50, resnet34



model = resnet34(pretrained=True)

class DualResNet(nn.Module):
    def __init__(self):
        super(DualResNet, self).__init__()
        self.model1 = resnet34(pretrained=True)
        self.model1.fc = nn.Identity()
        # self.model2.fc = nn.Identity()
        self.fc = nn.Sequential(
                    nn.Linear(1024,512),
                    nn.ReLU(),
                    nn.Linear(512,256),
                    nn.Dropout(0.3),
                    nn.ReLU(),
                    nn.Linear(256,128),
                    nn.ReLU(),
                    nn.Linear(128,2),
                    # nn.Sigmoid()
        )
    
    def forward(self, x, y):
        x = self.model1(x)
        y = self.model1(y)
        output = self.fc(torch.cat([x,y], 1))
        return output


if __name__ == "__main__":
    model_d = DualResNet()
    model_d.to("cuda")
    x = torch.rand((2,3,224,224), device="cuda")
    y = torch.rand((2,3,224,224), device="cuda")

    print(model_d(x,y).shape)