import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def build_model(output_size, weight_path=None):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, output_size)

    if weight_path:
        model.load_state_dict(torch.load(weight_path))

    return model