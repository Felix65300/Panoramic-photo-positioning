import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

from src.EfficientNet_B0.EfficientNet_B0_Feature_Fusion import _EfficientNet_MultiScale


# 1. 定義連續型 Hopfield 聯想記憶層
class _ContinuousHopfieldLayer(nn.Module):
    def __init__(self, feature_dim=1280, num_memories=512):
        super(_ContinuousHopfieldLayer,self).__init__()
        self.num_memories = num_memories
        self.feature_dim = feature_dim

        # =======================================
        # 建立可學習的記憶庫 (Memory Bank)
        # =======================================
        # Keys (鍵矩陣): 儲存特徵的索引
        # 用來與受損的輸入特徵進行比對
        self.keys = nn.Parameter(torch.randn(num_memories, feature_dim))

        # Values (值矩陣): 儲存乾淨的目標特徵
        # 匹配成功後輸出的內容
        self.values = nn.Parameter(torch.randn(num_memories, feature_dim))

        # 能量縮放因子 (Beta)，避免 Softmax 梯度消失
        self.beta = 1.0 / math.sqrt(feature_dim)

        # 初始化權重，確保初期記憶庫不會出現極端值
        nn.init.normal_(self.keys, mean=0.0, std=0.02)
        nn.init.normal_(self.values, mean=0.0, std=0.02)

        # 輸出前的正規化層，穩定特徵分佈
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        # x 形狀: [B, feature_dim] (例如 Batch Size x 1280)

        # 1. 計算輸入特徵與記憶庫的能量分數 (聯想過程)
        # 數學公式: scores = beta * (X * K^T)
        scores = torch.matmul(x, self.keys.t()) * self.beta # 形狀: [B, num_memories]

        # 2. 透過 Softmax 尋找能量谷底 (尋找最接近的記憶點)
        attn_weights = F.softmax(scores, dim=-1)

        # 3. 提取修復後的乾淨特徵
        # 數學公式: retrieved_memory = Softmax(beta * X * K^T) * V
        retrieved_memory = torch.matmul(attn_weights, self.values) # 形狀: [B, feature_dim]

        # 4. 殘差連接 (Residual Connection)
        # 將修復後的特徵與原特徵相加並正規化
        # 確保網路初期正常收斂
        out = self.layer_norm(x + retrieved_memory)
        return out

# 2. 模型定義
class _EfficientNet_B0_Hopfield(nn.Module):
    def __init__(self, num_classes=1000, num_memories=512):
        super(_EfficientNet_B0_Hopfield, self).__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # 拆除原生的分類頭
        # 在池化層之後介入
        self.backbone.classifier = nn.Identity()

        # 插入自定義的 Hopfield 記憶修復層
        self.hopfield = _ContinuousHopfieldLayer(feature_dim=1280, num_memories=num_memories)

        # 重建最終的分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # 1. 骨幹網路萃取特徵，並經過全局平均池化
        # (輸出形狀: [B, 1280])
        x = self.backbone(x)

        # 2. 將特徵送入 Hopfield 層進行聯想與修復
        x = self.hopfield(x)

        # 3. 將修復後的乾淨特徵送入分類器
        out = self.classifier(x)
        return out

def build_model(num_classes=1000):
    return _EfficientNet_B0_Hopfield(num_classes=num_classes, num_memories=512)

if __name__ == '__main__':
    model = build_model()
    print(model)