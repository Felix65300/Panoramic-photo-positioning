import numpy as np
from torchvision import transforms, datasets
from PIL import Image


# 提供一個函式讓外部取得標準的 Transform
def get_transforms(img_width, img_height, is_train=True):
    transform_list = []

    # 1. Resize
    transform_list.append(transforms.Resize((img_height, img_width)))

    # 2. 轉 Tensor (自動做 / 255.0 和 HWC->CHW)
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)

# ---------------------------------------------------
# 【關鍵】 取得 Dataset 的包裝函式
# 這裡把 ImageFolder 藏起來，外部只要呼叫這個函式就好
# ---------------------------------------------------
def get_train_dataset(root_dir, width, height, is_train=True):
    # 1. 取得對應的 Transform
    my_transform = get_transforms(width, height, is_train)

    # 2. 建立 ImageFolder
    # ImageFolder 會自動掃描 root_dir 下的所有子資料夾
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transform)
    return dataset
