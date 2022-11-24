import torch.nn as nn
from torchvision.models import resnet18

def resnet(num_classes=8) :
    model = resnet18(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=num_features, out_features=128, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=128, out_features=64, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=64, out_features=num_classes, bias=True))
    return model