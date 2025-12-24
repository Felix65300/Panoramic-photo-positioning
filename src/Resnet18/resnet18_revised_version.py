import torch.nn as nn
from torchvision.models import resnet18

def get_pano_model(num_classes=1000, pretrained=False):
    """
    統一管理模型的定義。
    
    Args:
        num_classes (int): 分類的類別數量，預設 1000。
        pretrained (bool): 是否使用 ImageNet 預訓練權重。
                           訓練時通常設為 False (或是 True 加速收斂)，
                           推論時一定要設為 False (因為我們要載入自己的權重)。
    Returns:
        model: 修改過最後一層的 ResNet18 模型
    """

    # 1. 根據參數決定是否載入官方預訓練權重
    weights = 'IMAGENET1K_V1' if pretrained else None
    
    # 2. 建立原始 ResNet18
    model = resnet18(weights=weights)
    
    # 3. 修改全連接層 (Fully Connected Layer)
    # 取得原始 fc 的輸入特徵數 (通常是 512)
    in_features = model.fc.in_features
    
    # 替換成我們指定的類別數
    model.fc = nn.Linear(in_features, num_classes)
    
    return model