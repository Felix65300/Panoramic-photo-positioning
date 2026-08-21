import torch
import torch.nn as nn
import math
import torchvision.models as models
from openpyxl.workbook import child
from torchvision.ops import SqueezeExcitation

# 1. 定義 ECA 模組 (Efficient Channel Attention)
class ECA(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        # 根據通道數自適應計算 1D 卷積的 kernel size (k_size 必須是奇數)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 使用 1D 卷積取代 SE 模組的兩層 FC
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        # =============================
        # ReZero (零初始化保護標量) 實作
        # =============================
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        # 空間全局平均池化
        y = self.avg_pool(x) # 形狀：[B, C, 1, 1]

        # 維度轉換以符合 1D 卷積的輸入格式
        y = y.squeeze(-1).transpose(-1,-2) # 形狀轉換為： [B, 1, C]

        # 透過 1D 卷積計算通道權重，並使用 Sigmoid 激勵
        y = self.conv(y)
        y = y.transpose(-1,-2).unsqueeze(-1) # 形狀轉換回：[B, C, 1, 1]

        mask = self.sigmoid(y)

        # 【加入 ReZero】
        # 當 alpha = 0 時，輸出完美等同於原特徵 x
        out = x + self.alpha * (x * mask - x)

        # 將計算出的通道權重乘回原始特徵圖
        return out

# 2. 定義遞迴替換函數
def replace_se_with_eca(module):
    """
    走訪模型的所有子模型，將 torchvision 原生的 SqueezeExcitation 替換為 ECA 模組
    """
    for name, child in module.named_children():
        # 若確認為 SE 模組，則進行替換
        if isinstance(child, SqueezeExcitation):
            # 從原生 SE 模組的第一層卷積獲取輸入通道數
            in_channels = child.fc1.in_channels
            # 將模組複寫為 ECA
            setattr(module, name, ECA(channels=in_channels))
        else:
            # 遞迴檢查下一層
            # 使用遞迴不使用迭代，EfficientNet本身是樹狀結構
            # Sequential 包著 MBConv，MBConv 裡面又包著 SqueezeExcitation
            # 使用迴圈走訪未知深度的樹狀結構還需要建立 Stack 或是 Queue 來寫 DFS 或 BFS
            replace_se_with_eca(child)

# 3. 模型定義
def build_model(num_classes=1000):
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    replace_se_with_eca(model)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes)
    )

    return model

if "__main__" == __name__:
    print(build_model())