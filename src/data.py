from torch.utils.data import Dataset
import cv2
import numpy as np
import os

from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, csv_data, img_dir, transform=None, is_train=True, target_size=(512,128)):
        self.data = csv_data
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 取得圖片名稱 (修正後的，移除引號)
        filename = row["filename"].strip("'")

        # 產生圖片完整路徑
        img_path = os.path.join(self.img_dir,filename )

        # 讀圖片
        image = cv2.imread(img_path)
        h,w,c = image.shape

        # # 改圖片，資料增強
        # if self.is_train:
        #     shift = np.random.randint(0,w)
        #     image = np.roll(image,shift,axis=1)

        # Resize & Normalize
        image = cv2.resize(image, self.target_size)
        # 讀 id
        id = row["id"]

        # toTensor
        image = self.to_tensor(image)

        return image, id, filename
