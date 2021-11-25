from torchvision.models import resnet50

model = resnet50(pretrained=True)


if __name__ == "__main__":
    print(model)