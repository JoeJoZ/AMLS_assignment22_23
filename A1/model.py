import torch
import torch.nn as nn
import torchvision.models as models

def modelA1():
    ResNet = models.resnet18(pretrained=True)
    fc_features = ResNet.fc.in_features
    ResNet.fc = nn.Linear(fc_features, 2)
    return ResNet


if __name__ == '__main__':
    x = torch.randn(2, 3, 218, 178).cuda()
    model = modelA1().cuda()
    y = model(x)
    print(y.shape)
