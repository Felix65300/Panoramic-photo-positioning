import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# data.py 跨資料夾，所以需要額外動作來輔助 import
# 1. 取得目前檔案的 (Training.py) 所在目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 取得上一層目錄 (專案的根目錄)
parent_dir = os.path.dirname(current_dir)
Project_Root = os.path.dirname(parent_dir)

# 3. 將根目錄加入系統搜尋路徑
sys.path.append(parent_dir)

# 4. 開始 import
from src.data import MyDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Convolution_Class import  CNN
import torchvision
import  matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# print(torch.cuda.is_available())

# ---------------------------------
# 1. 設定參數與裝置
# ---------------------------------
BATCH_SIZE = 16 # 根據顯卡記憶體調整 (16 或 32)
Learning_Rate = 0.001 # Adam 的標準學習率
Num_Epoch = 20


# ----------------------------------
# 2. 準備資料
# ----------------------------------

csv_path = os.path.join(Project_Root, "_gkcN1hzqm1RFcsvpk5Xmg", 'stitched_pano_final.csv')
img_path = os.path.join(Project_Root, "_gkcN1hzqm1RFcsvpk5Xmg")
df = pd.read_csv(csv_path)

'''
# -------------------------------------------------------
# 🔥 過擬合測試模式 (Overfit Test Mode)
# 目的：檢查程式邏輯有沒有寫錯，確認模型能不能「死記硬背」
# ---------------------------------------------------------

# 1. 先建立完整的 Dataset (跟原本一樣)
full_dataset = MyDataset(
    csv_data=pd.read_csv(csv_path),
    img_dir=img_path,
    is_train=True  # 先開著增強沒關係，強的模型應該也要能背起來
)

# 2. 【關鍵】只切出前 16 張圖片
# 使用 torch.utils.data.Subset
indices = range(16) # 取第 0 到第 15 張
train_dataset = torch.utils.data.Subset(full_dataset, indices)

print(f"⚠️ 正在進行過擬合測試！")
print(f"⚠️ 訓練資料數量: {len(train_dataset)} (原本是 1000)")

# 3. 【關鍵】DataLoader 設定
# shuffle=False: 不要亂數洗牌，題目順序固定，讓模型更好背
# batch_size=16: 一次就把這 16 張全看完
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
'''

train_dataset = MyDataset(df, img_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------------
# 3. 初始化模型、Loss、優化器
# ------------------------------------
cnn = CNN().to("cuda")

# loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to("cuda")
loss_func = nn.CrossEntropyLoss().to("cuda")
optimizer = optim.Adam(cnn.parameters(), lr=Learning_Rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode = 'min', factor = 0.1, patience = 3
)


# ------------------------------------
# 4. 開始訓練
# ------------------------------------
epoch_losses = []
if os.path.exists("pano_cnn_model.pth"):
    cnn.load_state_dict(torch.load('pano_cnn_model.pth'))

cnn.train()

for epoch in range(Num_Epoch):
    running_loss = 0.0

    # 使用 tqdm 顯示進度條
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Num_Epoch}", ncols = 100, leave = True) as loop:

        for images, labels, filenames in loop:
            # A. 搬移資料到 GPU
            images = images.to("cuda")
            labels = labels.to("cuda")

            # B. 歸零梯度
            optimizer.zero_grad()

            # C. Forward Pass
            outputs = cnn(images)

            # D. 計算 Loss
            loss = loss_func(outputs, labels)

            # E. Backward Pass
            loss.backward()

            # F. 更新參數
            optimizer.step()

            # --- 紀錄數據 ---
            running_loss += loss.item()

            # 進度調顯示即時 loss
            loop.set_postfix(loss=loss.item())


        # 印出這一個 Epoch 的平均 Loss
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} Result: Loss={avg_loss:.4f}")

# --------------------------
# 5. 畫出 Loss 折線圖
# --------------------------
plt.figure(figsize=(10, 5))
plt.plot(epoch_losses, label='Training Loss')
plt.title('Training Loss Trend')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.legend()
plt.grid(True)
plt.xticks(range(0, Num_Epoch, 2))

# 存檔而不是只有顯示 (方便在 Server 上看)
plt.savefig('loss_curve.png')
print("訓練結束！Loss 圖表已儲存為 loss_curve.png")

# 儲存模型權重
torch.save(cnn.state_dict(), 'pano_cnn_model.pth')
print("模型已儲存為 pano_cnn_model.pth")