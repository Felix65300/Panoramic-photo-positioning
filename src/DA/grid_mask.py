import math
import random
import torch
import  torch.nn as nn

class GridMask(nn.Module):
    """
    Grid Mask 資料增強：模擬環境中的遮蔽物
    d : 網格的週期 (重複單元的大小)
    ratio : 遮蓋面積占整張圖的比例 (預設 10%)
    """
    def __init__(self, d=32, ratio=0.1):
        super().__init__()
        self.d = d
        self.ratio = ratio

    def forward(self, img):
        # img: Tensor (C, H, W)
        _, _, h, w = img.size()

        # 1. 計算遮蓋邊長 r，確保總面積遮蓋率固定為 ratio
        # 公式推導：r = d * sqrt(ratio)
        self.r = int(self.d * math.sqrt(self.ratio))

        # 2. 建立遮罩基底 (全 1 代表保留，使用 device 確保 GPU 相容)
        mask = torch.ones((h,w), dtype=torch.float32, device=img.device)

        # 3. 生成隨機偏移量 (讓網格每次出現位置不同，避免模型記住位置)
        #delta_x = random.randint(0, self.d - 1)
        #delta_y = random.randint(0, self.d - 1)

        # 4. 挖洞邏輯 (將特定區域設為 0)
        # 使用切片 (Slicing) 取代迴圈，提升運算效率
        for y in range(delta_y - self.d, h, self.d):
            for x in range(delta_x - self.d, w, self.d):

                # 計算實際的起點與終點，並使用 max(0, ...) 防止索引為負數
                start_y = max(0, y)
                start_x = max(0, x)

                # 使用 min(..., h/w) 防止超出圖片右側與下方範圍
                end_y = min(y + self.r, h)
                end_x = min(x + self.r, w)

                # 確保計算出的區塊有效（起點小於終點）才進行遮罩
                if start_y < end_y and start_x < end_x:
                    mask[start_y:end_y, start_x:end_x] = 0.0

        # 5. 套用遮罩
        return  img * mask
