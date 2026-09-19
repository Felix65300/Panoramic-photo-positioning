import torch
import torch.nn as nn
import torch.fft
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class _FourierIlluminationAugmentation(nn.Module):
    """
    頻域亮度擴增：將影像轉換至頻域，擾動振幅(亮度)、保留相位(結構)
    再轉換回空間域
    """
    def __init__(self, perturbation_ratio=0.95):
        # 擾動比例：0.95 代表振幅會隨機縮放 5% ~ 195%
        super(_FourierIlluminationAugmentation, self).__init__()
        self.ratio = perturbation_ratio

    def forward(self, x):
        # 【關鍵】：只有在 model.train() 模式下才進行擾動
        # 在 model.eval() 測試階段，直接原封不動回傳 x
        if not self.training:
            return x

        # 1. 進行 2D 快速傅立葉轉換
        fft = torch.fft.fft2(x)
        # 將低頻(整體亮度分布)移到頻譜中心
        # x 的維度是 (B, C, H, W)
        # 隊最後兩個維度 (H, W) 進行 2D 傅立葉轉換
        fft_shifted = torch.fft.fftshift(fft,dim=(-2, -1))

        # 2. 分離振幅 (Amplitude) 與 相位 (Phase)
        amp = torch.abs(fft_shifted)
        phase = torch.angle(fft_shifted)

        # 3. 隨機擾動振幅 (模擬極端亮度變化)
        # 產生 Batch 級別的個隨機縮放係數 (係數介於 0.05 到 1.95 之間)
        # 維度設為 (B, 1, 1, 1) 以便利用廣播機制 (Broadcasting) 乘上特徵圖
        B = x.size(0)
        scale = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * self.ratio
        perturbed_amp = amp * scale

        # 4. 重新組合擾動後的振幅與原始相位
        perturbed_fft_shifted = perturbed_amp * torch.exp(1j * phase)

        # 5. 逆轉換回空間域影像
        fft_new = torch.fft.ifftshift(perturbed_fft_shifted, dim=(-2, -1))
        x_new = torch.fft.ifft2(fft_new).real

        # 確保像素值落在 0 到 1 之間
        return torch.clamp(x_new, 0.0, 1.0)

class _EfficientNet_B0_FDA(nn.Module):
    def __init__(self, num_classes=1000):
        super(_EfficientNet_B0_FDA, self).__init__()

        # 加裝亮度抗干擾層
        self.fda_layer = _FourierIlluminationAugmentation(perturbation_ratio=0.95)

        # 載入主幹網路
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        # 重建分類頭
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # 1. 經過頻域擾動層 (會自動根據 train/eval 決定是否啟動)
        x = self.fda_layer(x)

        # 2. 進入主幹網路提取特徵並分類
        out = self.backbone(x)

        return out

def build_model(num_classes=1000):
    print("Building EfficientNet_B0_FDA 🏗️")
    model = _EfficientNet_B0_FDA(num_classes)
    return model