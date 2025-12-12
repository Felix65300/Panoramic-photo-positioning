import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data import MyDataset
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.models import resnet18
import numpy as np

MODEL_SAVE_PATH = 'resnet18_pano_1000classes_optimized.pth'

# ✅ 新增：自定義全景圖滾動增強 (模擬相機旋轉)
class RandomRoll(object):
    def __init__(self, max_shift=0.5):
        self.max_shift = max_shift

    def __call__(self, img):
        # img 是 Tensor (C, H, W)
        _, _, w = img.shape
        shift = int(w * np.random.uniform(-self.max_shift, self.max_shift))
        return torch.roll(img, shifts=shift, dims=2)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 讀取 CSV
    csv_data = pd.read_csv("./_gkcN1hzqm1RFcsvpk5Xmg/stitched_pano_final.csv")
    
    # Transform 優化
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        
        # 1. 顏色抖動 (保持你原有的)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        # 2. 隨機裁切 (保持你原有的)
        transforms.RandomResizedCrop((128, 512), scale=(0.9, 1.0), ratio=(3.5, 4.5)), 
        
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

        # ✅ 3. 新增：全景圖專用滾動 (在 Normalize 之後做 Tensor 操作)
        RandomRoll(max_shift=0.5) 
    ])

    dataset = MyDataset(csv_data, "./_gkcN1hzqm1RFcsvpk5Xmg", transform)
    
    # ✅ 優化：num_workers 建議設為 CPU 核心數，batch_size 若顯存夠大可再加大
    trainloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, 
                             num_workers=8, pin_memory=True, persistent_workers=True)

    # 建立模型
    # ✅ 建議：如果不是接續訓練，請開啟預訓練權重 weights='IMAGENET1K_V1'
    model = resnet18(weights=None) 
    model.fc = nn.Linear(model.fc.in_features, 1000)

    # 嘗試載入舊權重 (如果存在)
    try:
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print("已載入先前的權重繼續訓練。")
    except FileNotFoundError:
        print("未找到權重檔，從頭開始訓練 (或從 ImageNet 權重開始)。")

    model = model.to(device)

    # ✅ 優化 Loss：加入 Label Smoothing (防止過擬合，讓 Loss 下降更平滑)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # ✅ 優化 Optimizer：改用 AdamW (收斂通常比 SGD 快且穩)
    # 若是接續訓練，lr 建議改小一點 (如 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # ✅ 優化 Scheduler：改用 Cosine Annealing (餘弦退火)
    # T_max=epochs 表示在訓練結束時 LR 降到最低
    epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print("Start Training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for img, id_label, *_ in trainloader:
            img, id_label = img.to(device), id_label.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            
            loss = criterion(outputs, id_label.long())
            loss.backward()
            
            # ✅ 新增：梯度裁剪 (防止梯度爆炸，讓 Loss 下降更穩)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += id_label.size(0)
            correct += (predicted == id_label).sum().item()

        # Step 更新放在 epoch 迴圈外
        scheduler.step()

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {current_lr:.6f}")
        
        # 保存策略：Loss 創新低或每 5 epoch 存一次
        if (epoch + 1) % 5 == 0 or avg_loss < 0.01:
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model Saved: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()