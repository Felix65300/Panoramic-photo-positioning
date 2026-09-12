import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.ops.misc import SqueezeExcitation


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

        self.alpha = nn.Parameter(torch.zeros(1))

        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        attention = self.sigmoid(out)

        return x * (1.0 - self.alpha) + (x * attention) * self.alpha

class CBAM(nn.Module):
    def __init__(self, se_module):
        super().__init__()
        self.cam = se_module
        self.sam = SpatialAttention()

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x

def replace_se_with_cbam(model):
    for name, module in model.named_children():
        if isinstance(module, SqueezeExcitation):
            setattr(model, name, CBAM(module))
        else:
            replace_se_with_cbam(module)

def build_model(num_classes):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    replace_se_with_cbam(model)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes)
    )

    return model

def __main__():
    model = build_model(1000)
    print(model)
if __name__ == '__main__':
    __main__()