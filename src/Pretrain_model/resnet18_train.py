import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from resnet18_revised_version import get_pano_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
Project_Root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from data import MyDataset



MODEL_SAVE_PATH =os.path.join(current_dir,'resnet18_pano_1000classes_optimized.pth')

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = os.path.join(Project_Root, "Dataset_Step1", 'stitched_pano_final.csv')
    img_path = os.path.join(Project_Root, "Dataset_Step1")
    df = pd.read_csv(csv_path)
    
    dataset = MyDataset(df, img_path)
    
    trainloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True, 
                             num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_pano_model(num_classes=1000, pretrained=False)

    try:
        # 讀取時也使用新的路徑
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Load weight: {MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print("Not found pth")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    
    epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    epoch_losses = []

    print("Start Training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, leave=True) as loop:
            for img, id_label, *_ in loop:
                img, id_label = img.to(device), id_label.to(device)

                optimizer.zero_grad()
                outputs = model(img)
                
                loss = criterion(outputs, id_label.long())
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                optimizer.step()

                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += id_label.size(0)
                correct += (predicted == id_label).sum().item()
                current_acc = 100 * correct / total

                loop.set_postfix(loss=loss.item(), acc=f"{current_acc:.2f}%")

        scheduler.step()

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1} Result: Loss={avg_loss:.4f} | Acc={epoch_acc:.2f}% | LR={current_lr:.6f}")

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
    print(f"訓練結束！Loss 圖表已儲存為 {plot_save_path}")
        
    # 儲存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model Saved: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()