import torch
import torch.nn as nn
import torchvision.models as models


def modelA2():
    ResNet = models.resnet18(pretrained=True)
    conv_channels = ResNet.conv1.out_channels
    ResNet.conv1 = nn.Conv2d(4, conv_channels, kernel_size=7, stride=2, padding=3, bias=False)
    fc_features = ResNet.fc.in_features
    ResNet.fc = nn.Linear(fc_features, 2)
    return ResNet


if __name__ == '__main__':
    x = torch.randn(2, 4, 218, 178).cuda()
    model = modelA2().cuda()
    y = model(x)
    print(y.shape)
