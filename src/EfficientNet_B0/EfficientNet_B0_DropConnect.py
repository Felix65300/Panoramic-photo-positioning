import torch.nn as nn
import torchvision.models as models

# =================================
# 變因四：正規化強度
# =================================

def build_model(num_classes):
    # 1. 載入 ImageNet 預訓練權重
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights, stochastic_depth_prob=0.3)

    # 2. 重建分類頭
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes)
    )

    return model