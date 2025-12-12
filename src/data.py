from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, csv_data, img_dir, transform=None):
        self.data = csv_data
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 取得圖片名稱 (修正後的，移除引號)
        filename = row["filename"].strip("'")

        # 產生圖片完整路徑
        img_path = os.path.join(self.img_dir,filename )

        # 讀圖片
        image = Image.open(img_path).convert("RGB")

        # 讀 id
        id = row["id"]

        # transform
        if self.transform:
            image = self.transform(image)

        return image, id
