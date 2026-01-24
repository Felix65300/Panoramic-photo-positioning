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

if __name__ == "__main__":
    print("Testing data.py integration...")