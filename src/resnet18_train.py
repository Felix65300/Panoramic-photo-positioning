import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from data import MyDataset
from torch.utils.data import DataLoader
import pandas as pd
from torchvision.models import resnet18

MODEL_SAVE_PATH = 'resnet18_pano_1000classes.pth'

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 讀取 CSV
    csv_data = pd.read_csv("./_gkcN1hzqm1RFcsvpk5Xmg/stitched_pano_final.csv")
    
    # 檢查一下 Label 的最大值，確保不會爆掉
    max_label = csv_data.iloc[:, 1].max() # 假設第二欄是 Label
    print(f"數據集中的最大 Label ID 為: {max_label}")
    if max_label >= 1000:
        print("⚠️ 警告：Label ID 超過 999，程式可能會報錯！請確保 ID 是 0-999。")

    # 全景圖建議 Transform (高 128, 寬 512)
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        # 隨機水平翻轉 (全景圖如果是有方向性的，這個要斟酌使用，或是使用 Roll)
        # transforms.RandomHorizontalFlip(p=0.5), 
        
        # 關鍵：隨機改變亮度、對比、飽和度，模擬不同光影
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        # 關鍵：隨機微幅裁切再放大 (模擬鏡頭遠近或位置偏差)
        transforms.RandomResizedCrop((128, 512), scale=(0.9, 1.0), ratio=(3.5, 4.5)), 
        
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 測試用的 Transform (保持原樣)
    val_transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = MyDataset(csv_data, "./_gkcN1hzqm1RFcsvpk5Xmg", transform)
    
    trainloader = DataLoader(dataset=dataset, batch_size=128, shuffle=False,num_workers=4, pin_memory=True,persistent_workers=True)

    # 建立模型
    model = resnet18(weights=None)
    
    # ✅ 關鍵修改：輸出層設為 1000
    model.fc = nn.Linear(model.fc.in_features, 1000)

    #載入自己的權重檔案
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device) 

    model.load_state_dict(checkpoint)

    model = model.to(device)

    # 定義 Loss 和 Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    epochs = 50
    print("Start")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # ✅ 使用 *_ 處理多餘的回傳值
        for img, id_label, *_ in trainloader:
            img, id_label = img.to(device), id_label.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            
            # id_label 必須是 long type (整數)
            loss = criterion(outputs, id_label.long()) 
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # 計算準確率
            _, predicted = torch.max(outputs.data, 1)
            total += id_label.size(0)
            correct += (predicted == id_label).sum().item()

        scheduler.step()

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {current_lr:.5f}")
        
        # 1000 類別比較難練，Loss 可能要降到更低才算好
        if epoch_acc >= 99.5 and avg_loss < 0.01:
            print("模型收斂完畢，提早結束。")
            break

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Save: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()