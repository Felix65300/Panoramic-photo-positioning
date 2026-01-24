import torch.nn as nn
from torchvision import transforms

class RandomBrightness(nn.Module):
    """
    隨機調整料度與對比度
    封裝 torchvision.transforms.ColorJitter 以保持介面一致
    """
    def __init__(self, brightness = 0.2, contrast = 0.2):
        super().__init__()
        # brightness = 0.2 代表亮度在 [0.8, 1.2] 之間隨機變化
        # brightness 設定的是正負的範圍 (例如：正負0.2)
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)

    def forward(self, img):
        return self.jitter(img)