import os
import sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# data_Step2.py 跨資料夾，所以需要額外動作來輔助 import
# 1. 取得目前檔案的 (Training.py) 所在目錄
current_dir = Path.cwd()

# 2. 取得上一層目錄 (專案的根目錄)
Custom_model = os.path.dirname(current_dir)
src = os.path.dirname(Custom_model)
Project_Root = os.path.dirname(src)

# 3. 將根目錄加入系統搜尋路徑
sys.path.append(Custom_model)
sys.path.append(src)
sys.path.append(Project_Root)

# 4. 開始 import

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_Step2_5 import get_dataset
from src.Custom_model.Convolution_Class import CNN

# ---------------------------------
# 1. 設定參數與裝置
# ---------------------------------
BATCH_SIZE = 128 # 根據顯卡記憶體調整 (16 或 32)
Learning_Rate = 1e-4 # Adam 的標準學習率
Num_Epoch = 200
IMG_WIDTH = 512
IMG_HEIGHT = 128
DEVICE = torch.device('cuda')
TRAIN_DIR = Project_Root + '/Datasets/Dataset_Step1'
TEST_ROOT = Project_Root + '/Datasets/Dataset_Step2'
MODEL_PATH = 'Step2.5_pano_cnn_model.pth'
DA_ACCURACY = {'Brightness':list(),
               'Colortemperature':list(),
               'Grid_Mask':list(),
               'Horizontal_Roll':list()}

def model_test(model):
    da_conditions = {'Brightness': list(range(0, 210, 10)),
                     'Colortemperature': list(range(0, 210, 10)),
                     'Grid_Mask': list(range(0, 110, 10)),
                     'Horizontal_Roll': list(range(0, 370, 10))}
    for category, values in da_conditions.items():
        for val in values:
            test_dir = TEST_ROOT + f'/{category}/{val}'
            if category == 'Horizontal_Roll':
                test_dir += '°'
            else :
                test_dir += '%'
            print(test_dir)
            test_dataset = get_dataset(test_dir, IMG_WIDTH, IMG_HEIGHT, is_train=False)

            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="Testing", unit='batch'):
                    images, labels = images.to(DEVICE), labels.to(DEVICE)

                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    DA_ACCURACY[category].append(100 * correct / total)

def model_training ():
    # ----------------------------------
    # 2. 準備資料 (呼叫 data_Step1.py)
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
    best_loss = float('inf')

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint['best_loss']


    model.train()
    print("--> 開始訓練...")
    running_loss = 0.0

    with tqdm(train_loader, ncols=100) as loop:
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
    scheduler.step(avg_loss)

    print(f"Loss: {avg_loss:.4f} | LR: {current_lr:.8f}")


    if avg_loss < best_loss:
        best_loss = avg_loss

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
        }

        torch.save(checkpoint, MODEL_PATH)
    model_test(model)

def main():
    model_training()
if __name__ == '__main__':
    main()