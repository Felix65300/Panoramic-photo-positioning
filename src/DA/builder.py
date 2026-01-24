from torchvision import transforms

# 引用同資料夾下的自定義模組
from .grid_mask import GridMask
from .color_temperature import RandomColorTemperature
from .brightness import RandomBrightness

def get_transforms(img_width = 512, img_height=128, is_train=True):
    """
    工廠函數：根據需求組裝所有資料增強模組
    """
    transform_list = []

    # 1. 基礎處理：縮放 + 轉 Tensor
    transform_list.append(transforms.Resize((img_height, img_width)))
    transform_list.append(transforms.ToTensor())

    # 2. 訓練階段才做的資料增強 (Data Augmentation
    if is_train:
        # --- A. 亮度與對比度 ---
        transform_list.append(RandomBrightness(brightness=0.2,contrast=0.2))

        # --- B. 色溫調整 ---
        transform_list.append(RandomColorTemperature(range_temp=(0.8,1.2)))

        # --- C. 網格遮罩 ---
        # 固定 10% 遮罩率，週期為 32
        transform_list.append(GridMask(d = 32, ratio=0.1))

    # 3. 標準化 (讓模型訓練數值更穩定)
    # 使用 ImageNet 的標準平均值與標準差
    transform_list.append(transforms.Normalize(mean=[0.485,0.456,0.406],
                                               sts=[0.229,0.224,0.225]))

    # 將列表組合成一個 transform popline
    return transforms.Compose(transform_list)