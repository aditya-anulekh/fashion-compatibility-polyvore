import torch.nn as nn
import torch
from torchvision.models import resnet50, resnet34, resnet18



model = resnet18(pretrained=True)

class DualResNet(nn.Module):
    def __init__(self):
        super(DualResNet, self).__init__()
        self.model1 = resnet18(pretrained=True)
        self.model1.fc = nn.Identity()
        # self.model2.fc = nn.Identity()
        self.fc = nn.Sequential(
                    nn.Linear(1024,256),
                    nn.ReLU(),
                    # nn.Linear(256,64),
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
    model_d = DualResNet()
    model_d.to("cuda")
    x = torch.rand((2,3,224,224), device="cuda")
    y = torch.rand((2,3,224,224), device="cuda")

    print(model_d(x,y).shape)