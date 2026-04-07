import torch.nn as nn
import torchvision.transforms.functional as F

class RandomBrightness(nn.Module):
    """
    隨機調整料度
    """
    def __init__(self, brightness = 0.2):
        super().__init__()
        # brightness: 亮度調整幅度 (預設 0.2 -> 0.8~1.2)
        # contrast: 對比度調整幅度 (預設 0.2 -> 0.8~1.2)
        self.brightness_factor = brightness

    def forward(self, img):
        return F.adjust_brightness(img, self.brightness_factor)