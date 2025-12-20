import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

# 環境變數設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
Project_Root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(Project_Root)

# Import 自定義模組
from src.data import get_dataset
from resnet18_revised_version import get_pano_model

IMG_WIDTH = 512
IMG_HEIGHT = 128
BATCH_SIZE = 128
epochs = 50

# 權重檔案名稱
MODEL_PATH = os.path.join(current_dir, 'resnet18_pano_1000classes_optimized.pth')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    csv_path = os.path.join(Project_Root, "Dataset_Step1", 'stitched_pano_final.csv')
    img_path = os.path.join(Project_Root, "Dataset_Step1")
    df = pd.read_csv(csv_path)
    
    # 1. 載入 Dataset
    dataset = get_dataset(root_dir=img_path, width=IMG_WIDTH, height=IMG_HEIGHT, is_train=True)
    
    # DataLoader
    trainloader = DataLoader(
        dataset=dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True, 
        persistent_workers=True
    )

    # 2. 定義模型、損失函數、優化器、學習率調整器
    model = get_pano_model(num_classes=1000, pretrained=False)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0) # weight_decay=0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 3. 嘗試載入先前的權重 (Resume)
    start_epoch = 0
    best_acc = 0.0 # 用來記錄歷來最高準確率

    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading weights from {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint) 
            print("Weight loaded successfully.")
        except Exception as e:
            print(f"Loading failed: {e}, training from scratch.")
    else:
        print("No existing weights found. Training from scratch.")

    epoch_losses = []

    print("Start Training...")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用 tqdm 顯示進度
        with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, leave=True) as loop:
            for img, id_label in loop:
                img, id_label = img.to(device), id_label.to(device)

                # 1. 梯度歸零
                optimizer.zero_grad()
                # 2. Forward
                outputs = model(img)
                loss = criterion(outputs, id_label)
                # 3. Backward
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                # 4. Update
                optimizer.step()

                # --- 統計 ---
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += id_label.size(0)
                correct += (predicted == id_label).sum().item()
                
                #即時顯示
                current_acc = 100 * correct / total
                loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

        # 更新 Learning Rate
        scheduler.step()

        # 計算 Epoch 結果
        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1} Result: Loss={avg_loss:.4f} | Acc={epoch_acc:.2f}% | LR={current_lr:.6f}")

        # 存取最佳權重 ---
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"★ New Best Model Saved! (Acc: {best_acc:.2f}%) saved to {MODEL_PATH}")
        
        # 移除了原本無條件儲存的程式碼

    # 繪圖部分
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.title('Training Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    
    if epochs > 20:
        plt.xticks(range(0, epochs, 5)) 
    else:
        plt.xticks(range(0, epochs, 1))

    plot_save_path = os.path.join(current_dir,'resnet18_loss_curve.png')
    plt.savefig(plot_save_path)
    print(f"Training finished! Loss chart saved as {plot_save_path}")

if __name__ == '__main__':
    main()