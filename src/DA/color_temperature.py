import torch
import torch.nn as nn

class RandomColorTemperature(nn.Module):
    """
    隨機調整色溫 (模擬冷暖色調變化)
    range_temp: 調整因子範圍，預設 (0.8, 1.2)
        - factor > 1.0: 變暖 (紅多藍少)
        - factor < 1.0: 變暖 (紅少藍多)
    """
    def __init__(self, ratio = 2.0):
        super().__init__()
        self.ratio = ratio

    """
    要做這個的時候記得把DA_Dataset_Frame裡的aug_tensor = api(img_tensor.unsqueeze(0)).squeeze(0)
    改成aug_tensor = api(img_tensor)，因為此API是直接處理 (C, H, W)，不是(B, C, H, W)
    """
    def forward(self, img):
        # 建立一個完全獨立的張量複本，保護原圖不被破壞！
        img = img.clone()
        # 決定這次的調整因子
        img[0] = img[0] * self.ratio # R 通道
        img[2] = img[2] * (2-self.ratio) # B 通道

        return torch.clamp(img, 0.0,1.0)