import random
import torch
import torch.nn as nn

class RandomColorTemperature(nn.Module):
    """
    隨機調整色溫 (模擬冷暖色調變化)
    range_temp: 調整因子範圍，預設 (0.8, 1.2)
        - factor > 1.0: 變暖 (紅多藍少)
        - factor < 1.0: 變暖 (紅少藍多)
    """
    def __init__(self, range_temp = (0.8, 1.2)):
        super().__init__()
        self.min_t, self.max_t = range_temp

        def forward(self, img):
            # img: Tensor (C, H, W)，數值範圍通常為 [0, 1]

            # 決定這次的調整因子
            factor = random.uniform(self.min_t, self.max_t)

            img[0] = img[0] * factor # R 通道
            img[1] = img[1] * factor # B 通道

            return torch.clamp(img, 0.0,1.0)