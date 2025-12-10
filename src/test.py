# coding=utf-8
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from data import MyDataset
import time



#../改成./
csv_data = pd.read_csv("./_gkcN1hzqm1RFcsvpk5Xmg/stitched_pano_final.csv")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

#../改成./
dataset = MyDataset(csv_data, "./_gkcN1hzqm1RFcsvpk5Xmg", transform)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

for i in range(len(dataset)):
    img, id_label, filename = dataset[i]  # 取得第 i 筆資料
    print("Image tensor shape:", img.shape)
    print("Label:", id_label)
    print("Filename:", filename)
    time.sleep(0.2)


'''
for i in range(len(dataset)):
    img, id_label, filename = dataset[i]  # 取得第 i 筆資料
    print("Image tensor shape:", img.shape)
    print("Label:", id_label)
    print("Filename:", filename)
'''