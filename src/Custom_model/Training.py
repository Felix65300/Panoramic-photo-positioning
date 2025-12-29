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
sys.path.append(Project_Root)

# 4. 開始 import

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import  matplotlib.pyplot as plt
from tqdm import tqdm
from src.data import get_dataset
from Convolution_Class import CNN

# ---------------------------------
# 1. 設定參數與裝置
# ---------------------------------
BATCH_SIZE = 16 # 根據顯卡記憶體調整 (16 或 32)
Learning_Rate = 1e-4 # Adam 的標準學習率
Num_Epoch = 200
IMG_WIDTH = 512
IMG_HEIGHT = 128
DEVICE = torch.device('cuda')
TRAIN_DIR = os.path.join(Project_Root, "Dataset_Step1")
MODEL_PATH = 'pano_cnn_model.pth'

def model_training ():
    # ----------------------------------
    # 2. 準備資料 (呼叫 data.py)
    # ----------------------------------
    train_dataset = get_dataset(TRAIN_DIR,IMG_WIDTH,IMG_HEIGHT,is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---------------------------------------------------
    # 3. 初始化模型
    # ---------------------------------------------------
    model = CNN(num_classes=len(train_dataset.classes)).to(DEVICE)
    loss_func = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min', factor=0.5,patience = 10,min_lr = 1e-6
        )

    # ---------------------------------------------------
    # 4. 開始訓練
    # ---------------------------------------------------
    epoch_losses = []
    best_loss = float('inf')
    #
    # if os.path.exists(MODEL_PATH):
    #     checkpoint = torch.load(MODEL_PATH)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     best_loss = checkpoint['best_loss']
    #

    model.train()
    print("--> 開始訓練...")

    for epoch in range(Num_Epoch):
        running_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{Num_Epoch}",ncols=100) as loop:
            for images, labels in loop:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                current_lr = optimizer.param_groups[0]['lr']

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | LR: {current_lr:.8f}")

        #
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #
        #     checkpoint = {
        #         'model_state_dict': model.state_dict(),
        #         'best_loss': best_loss,
        #         'epoch': epoch+1
        #     }
        #
        #     torch.save(checkpoint, MODEL_PATH)
        #

    return epoch_losses

def main():
    epoch_losses = model_training()
    # 存圖表
    plt.figure(figsize=(10,5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.grid(True)
    plt.savefig('Loss_curve.png')

if __name__ == '__main__':
    main()