import torch.nn as nn
from torchvision.models import mobilenet_v3_small

def mobilenet(num_classes=8) :
    model = mobilenet_v3_small(weights='IMAGENET1K_V1')
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features=num_features, out_features=64, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=64, out_features=32, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features=32, out_features=num_classes, bias=True))
    return model