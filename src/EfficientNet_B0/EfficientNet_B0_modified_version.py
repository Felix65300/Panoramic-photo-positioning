import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def build_model(num_classes):
    # 加載預訓練的 EfficientNet_B0
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # EfficientNet_B0 的 classifier 結構為:
    # (0): Dropout(p=0.2, inplace=True)
    # (1): Linear(in_features=1280, out_features=1000, bias=True)

    # 獲取最後一層全連接層的輸入特徵數 (通常是 1280)
    in_features = model.classifier[1].in_features

    # 替換最後一層為新的分類層
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model