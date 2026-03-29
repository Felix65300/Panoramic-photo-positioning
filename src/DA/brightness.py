import torch.nn as nn
from torchvision import transforms

class RandomBrightness(nn.Module):
    """
    隨機調整料度與對比度
    封裝 torchvision.transforms.ColorJitter 以保持介面一致
    """
    def __init__(self, brightness = 0.2):
        super().__init__()
        # brightness: 亮度調整幅度 (預設 0.2 -> 0.8~1.2)
        # contrast: 對比度調整幅度 (預設 0.2 -> 0.8~1.2)
        self.jitter = transforms.ColorJitter(brightness=brightness)

    def forward(self, img):
        return self.jitter(img)