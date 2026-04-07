import os
from torchvision import datasets
from torch.utils.data import DataLoader

# 🔥 這裡只要引用 DA 模組就好
try:
    from src.DA import get_transforms
except ImportError:
    from DA import get_transforms

def get_dataset(root_dir, width=512, height=128, is_train=True):
    """
    建立資料集
    """
    # 1. Builder
    my_transform = get_transforms(img_width=width, img_height=height, is_train=is_train)

    # 2. 建立 ImageFolder
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transform)

    return dataset

def get_sample():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # 1. 設定測試參數
    # 請換成你電腦裡隨便一個有圖片的資料夾路徑，或者你的 Dataset_Step1 路徑
    # 假設你的專案結構，我們試著抓上一層的 Dataset
    TEST_ROOT = r"../Dataset_Step1"  # 👈 請依你的實際路徑修改

    # 防呆：如果路徑不存在，就不要跑
    if not os.path.exists(TEST_ROOT):
        print(f"❌ 找不到路徑: {TEST_ROOT}，請修改程式碼中的 TEST_ROOT")
    else:
        print(f"🔍 開始檢查資料增強效果，讀取路徑: {TEST_ROOT}")

        # 2. 建立訓練集 (is_train=True 代表會套用所有增強)
        # 我們故意設 batch_size=4 來抓幾張圖看
        dataset = get_dataset(root_dir=TEST_ROOT, width=512, height=128, is_train=True)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # 3. 定義「反標準化」函數 (把 Tensor 變回人類看得懂的圖片)
        # 因為我們之前做了 (x - mean) / std，現在要 (x * std) + mean
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # 4. 抓取一個 Batch 的資料
        images, labels = next(iter(loader))

        # 5. 設定畫布 (一橫排顯示 4 張圖)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle("Data Augmentation Preview (Felix's Dataset)", fontsize=16)

        for i in range(4):
            # 取出單張 Tensor 並轉為 Numpy
            img = images[i].numpy().transpose((1, 2, 0))  # [C, H, W] -> [H, W, C]

            # 執行反標準化： $Original = (Tensor \times std) + mean$
            img = std * img + mean

            # 修正數值範圍，避免因為浮點數誤差導致超出 [0, 1] 產生警告
            img = np.clip(img, 0, 1)

            # 繪圖
            axes[i].imshow(img)
            axes[i].set_title(f"Label: {labels[i].item():.2f}")  # 假設標籤是連續值
            axes[i].axis('off')  # 隱藏座標軸讓畫面乾淨

        # 6. 關鍵：直接跳出視窗預覽
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("Testing data.py integration...")


# 測試版 main (會有增強後的成果預覽圖)，使用時將上面的 main 註解，並把 28 行開始解除註解即可使用
# --- 測試與視覺化區塊 ---
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torch
#
#     # 1. 設定測試參數
#     # 請換成你電腦裡隨便一個有圖片的資料夾路徑，或者你的 Dataset_Step1 路徑
#     # 假設你的專案結構，我們試著抓上一層的 Dataset
#     TEST_ROOT = r"../Dataset_Step1"
#
#     # 防呆：如果路徑不存在，就不要跑
#     if not os.path.exists(TEST_ROOT):
#         print(f"❌ 找不到路徑: {TEST_ROOT}，請修改程式碼中的 TEST_ROOT")
#     else:
#         print(f"🔍 開始檢查資料增強效果，讀取路徑: {TEST_ROOT}")
#
#         # 2. 建立訓練集 (is_train=True 代表會套用所有增強)
#         # 我們故意設 batch_size=4 來抓幾張圖看
#         dataset = get_dataset(root_dir=TEST_ROOT, width=512, height=128, is_train=True)
#         loader = DataLoader(dataset, batch_size=4, shuffle=True)
#
#         # 3. 定義「反標準化」函數 (把 Tensor 變回人類看得懂的圖片)
#         # 因為我們之前做了 (x - mean) / std，現在要 (x * std) + mean
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#
#
#         def imshow(tensor_img, title=None):
#             # 轉成 Numpy: (C, H, W) -> (H, W, C)
#             img = tensor_img.numpy().transpose((1, 2, 0))
#
#             # 反標準化 (Un-normalize)
#             img = std * img + mean
#
#             # 確保數值在 0~1 之間 (因為浮點數運算可能有極小誤差)
#             img = np.clip(img, 0, 1)
#
#             return img
#
#
#         # 4. 抓一個 Batch 出來顯示
#         data_iter = iter(loader)
#         images, labels = next(data_iter)
#
#         # 5. 畫圖並存檔
#         fig, axes = plt.subplots(len(images), 1, figsize=(10, 8))
#         if len(images) == 1: axes = [axes]  # 防呆
#
#         print(f"📸 正在生成預覽圖...")
#         for idx, img in enumerate(images):
#             ax = axes[idx]
#             # 顯示圖片
#             restored_img = imshow(img)
#             ax.imshow(restored_img)
#             ax.axis('off')
#             ax.set_title(f"Augmented Sample {idx + 1}")
#
#         plt.show()
        # 存成檔案讓你看
        # save_path = "check_augmentation.jpg"
        # plt.tight_layout()
        # plt.savefig(save_path)
        # print(f"✅ 檢查完成！圖片已儲存為：{save_path}")
        # print("請打開這張圖片，確認有沒有看到：\n1. 黑色網格 (GridMask)\n2. 圖片左右平移 (HorizontalRoll)\n3. 顏色偏冷或偏暖 (ColorTemp)")