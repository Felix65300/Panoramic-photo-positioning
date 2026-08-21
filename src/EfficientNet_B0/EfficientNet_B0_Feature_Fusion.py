# 此模型實做的是 Multi-Scale Feature Fusion (多尺度特徵融合)
import torch
import torch.nn as nn
import torchvision.models as models

class _EfficientNet_MultiScale(nn.Module):
    def __init__(self, num_classes=1000):
        super(_EfficientNet_MultiScale,self).__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)

        # ===================================
        # 拆解 EfficientNet 的特徵萃取區塊 (總共 9 個 Block)
        # ===================================

        # 階段一：淺層特徵 (包含 Stem 與前三個 MBConv)
        # 提取所以 0, 1, 2, 3，輸出通道為 40 (保留強烈邊緣)
        self.stage1 = backbone.features[:4]

        # 階段二：中層特徵 (包含中間兩個 MBConv)
        # 提取所以 4, 5，輸出通道為 112 (保留局部形狀)
        self.stage2 = backbone.features[4:6]

        # 階段三：深層特徵 (包含最後三個 Block)
        # 提取所以 6, 7, 8，輸出通道為 1280
        self.stage3 = backbone.features[6:]

        # =====================================
        # 重建分類頭部
        # =====================================
        # 拚街後的特徵維度 = 40 + 112 + 1280 = 1430
        in_features = 40 + 112 + 1280

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2,inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # 1. 影像進入階段一，獲取淺層特徵
        f1 = self.stage1(x)

        # 2. 淺層特徵進入階段二，獲取中層特徵
        f2 = self.stage2(f1)

        # 3. 中層特徵進入階段三，獲取深層特徵
        f3 = self.stage3(f2)

        # 4. 對三個層級的特徵分別進行池化，並展平 (Flatten) 為 1D 向量
        p1 = torch.faltten(self.pool(f1), 1) # 形狀：[B, 40]
        p2 = torch.faltten(self.pool(f2), 1) # 形狀：[B, 112]
        p3 = torch.faltten(self.pool(f3), 1) # 形狀：[B, 1280]

        # 5. 在通道維度 (dim=1) 上將三者拼接
        f_cat = torch.cat([p1, p2, p3], dim=1) # 總長度: 1432

        # 6. 送出分類頭
        out = self.classifier(f_cat)
        return out

def build_model(num_classes=1000):
    return _EfficientNet_MultiScale(num_classes)

if __name__ == '__main__':
    model = build_model(1000)
    print(model)
