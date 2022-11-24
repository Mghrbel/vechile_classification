import torch.nn as nn
from torchvision.models import vgg16

def vgg(num_classes=8) :
    model = vgg16(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features=num_features, out_features=128, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=128, out_features=64, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=64, out_features=num_classes, bias=True))
    return model