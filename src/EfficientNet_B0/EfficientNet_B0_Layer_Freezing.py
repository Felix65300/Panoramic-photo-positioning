import torch
import torch.nn as nn
import torchvision.models as models


# ==================================
# 變因二：權重凍結 (Layer Freezing)
# ==================================

def build_model(num_classes):
    # 1. 載入 ImageNet 預訓練的 EfficientNet-B0
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # 2. 重建分類頭
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes)
    )

    # 3. 實作凍結策略
    # 設定凍結範圍：凍結 Stem 層與前 3 個 MBConv 區塊 (索引 0, 1, 2, 3)
    freeze_until_index = 4

    for idx, child in enumerate(model.features):
        if idx < freeze_until_index:
            for param in child.parameters():
                param.requires_grad = False

def __main__():
    model = build_model(1000)
    print(model)

if __name__ == '__main__':
    __main__()
