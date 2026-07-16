import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def build_model(num_classes):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model