import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def build_model(num_classes):
    # 載入預訓練權重
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # EfficientNet_B0 的 classifier 結構為:
    # (0): Dropout(p=0.2, inplace=True)
    # (1): Linear(in_features=1280, out_features=1000, bias=True)

    # 獲取最後一層全連接層的輸入特徵數 (1280)
    in_features = model.classifier[1].in_features

    # 替換最後一層為新的分類層
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    return model