import numpy as np
from torchvision import transforms, datasets
from PIL import Image

# ---------------------------------------------------
# 【關鍵】 取得 Dataset 的包裝函式
# 這裡把 ImageFolder 藏起來，外部只要呼叫這個函式就好
# ---------------------------------------------------
def get_test_dataset(root_dir):

    # 1. 取得對應的 Transform，由於測試資料集的圖片大小已經改成 512 * 128，不用重新 resize
    # 所以將 get_transform 刪掉直接 ToTensor 就好
    my_transform = transforms.Compose([transforms.ToTensor()])

    # 2. 建立 ImageFolder
    # ImageFolder 會自動掃描 root_dir 下的所有子資料夾
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transform)
    return dataset
