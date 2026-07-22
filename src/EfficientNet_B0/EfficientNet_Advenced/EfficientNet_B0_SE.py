import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 1. 定義 GeM (Generalized Mean Pooling) 層，以取代 GAP
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        # 將 p 設為可學習參數 (Learnable Parameter)，初始值設為 3.0
        # 使用 nn.Parameter 是因為模型進行反向傳播（Backpropagation）時，優化器只會去尋找被註冊為 nn.Parameter 的變數來計算梯度。
        # 加上 nn.Parameter，就等於告訴 PyTorch：「請把 p 當作跟卷積層權重一樣的變數，在每一次迭代中根據 Loss 去更新它。」
        # 使用 torch.ones(1) 是因為 nn.Parameter 只能包裝 Tensor
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # 將特徵圖數值限制在 eps 以上避免梯度出現除以 0 的情況，計算 p 次方
        x_clamp = x.clamp(min=self.eps).pow(self.p)
        # 進行全局平均池化
        x_pool = F.avg_pool2d(x_clamp, (x.size(-2), x.size(-1)))
        # 開 p 次方根後回傳
        return x_pool.pow(1. / self.p)
def build_model(num_classes):
    # 2. 載入 ImageNet 預訓練的 EfficientNet-B0
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # 3. 替換池化層 (將原本的 AdaptiveAvgPool2d 換成 GeM)
    model.avgpool = GeM()

    # 4. 重建分類頭 (Classifier Head)
    num_classes = num_classes
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
