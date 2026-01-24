from torchvision import transforms

# 引用同資料夾下的自定義模組
from .grid_mask import GridMask
from .color_temperature import RandomColorTemperature
from .brightness import RandomBrightness
from .horizontal_roll import RandomHorizontalRoll

def get_transforms(img_width = 512, img_height=128, is_train=True):
    """
    工廠函數：根據需求組裝所有資料增強模組
    """
    transform_list = []

    # 1. Resize (PIL -> PIL)
    transform_list.append(transforms.Resize((img_height, img_width)))

    # 2. 訓練階段的 PIL 層級增強 (Data Augmentation)
    if is_train:
        # 🔥 環景圖核心增強：左右平移
        # 放在 ToTensor 之前，因為它是針對像素矩陣操作
        transform_list.append(RandomHorizontalRoll())

    # 3. 轉 Tensor (PIL -> Tensor, 0~1)
    transform_list.append(transforms.ToTensor())

    # 4. 訓練階段 Tensor 層級增強 (Data Augmentation)
    if is_train:
        # --- A. 亮度與對比度 ---
        transform_list.append(RandomBrightness(brightness=0.2,contrast=0.2))

        # --- B. 色溫調整 ---
        transform_list.append(RandomColorTemperature(range_temp=(0.8,1.2)))

        # --- C. 網格遮罩 ---
        # 固定 10% 遮罩率，週期為 32
        transform_list.append(GridMask(d = 32, ratio=0.1))

    # 5. 標準化 (讓模型訓練數值更穩定)
    # 使用 ImageNet 的標準平均值與標準差
    transform_list.append(transforms.Normalize(mean=[0.485,0.456,0.406],
                                               sts=[0.229,0.224,0.225]))

    # 將列表組合成一個 transform popline
    return transforms.Compose(transform_list)