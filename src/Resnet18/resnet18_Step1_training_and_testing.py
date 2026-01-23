import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 路徑設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
Project_Root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(Project_Root)

from src.data import get_dataset
from resnet18_revised_version import get_pano_model

IMG_WIDTH = 512
IMG_HEIGHT = 128
BATCH_SIZE = 128
DEVIVE = torch.device("cuda")
IMG_PATH = os.path.join(Project_Root, "Dataset_Step1")
epochs = 200

MODEL_PATH = os.path.join(current_dir, 'resnet18_pano_1000classes_optimized.pth')

def resnet18_training_and_testing():
    dataset = get_dataset(root_dir=IMG_PATH, width=IMG_WIDTH, height=IMG_HEIGHT, is_train=True)

    trainloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = get_pano_model(num_classes=1000, pretrained=False)
    model = model.to(DEVIVE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    start_epoch = 0
    best_acc = 0.0

    # if os.path.exists(MODEL_PATH):
    #     try:
    #         print(f"Loading weights from {MODEL_PATH}")
    #         checkpoint = torch.load(MODEL_PATH, map_location=device)
    #         model.load_state_dict(checkpoint)
    #         print("Weight loaded successfully.")
    #     except Exception as e:
    #         print(f"Loading failed: {e}, training from scratch.")
    # else:
    #     print("No existing weights found. Training from scratch.")

    epoch_losses = []
    epoch_accs = []

    print("Resnet 18 Start training...")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, leave=False) as loop:
            for img, id_label in loop:
                img, id_label = img.to(DEVIVE), id_label.to(DEVIVE)

                optimizer.zero_grad()

                outputs = model(img)
                loss = criterion(outputs, id_label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += id_label.size(0)
                correct += (predicted == id_label).sum().item()

                current_acc = 100 * correct / total
                loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.2f}%")

        scheduler.step()

        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        epoch_accs.append(epoch_acc)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1} Result: Loss={avg_loss:.4f} | Acc={epoch_acc:.2f}% | LR={current_lr:.6f}")

        # 存取最佳權重 ---
        # if epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     torch.save(model.state_dict(), MODEL_PATH)
        #     print(f"★ New Best Model Saved! (Acc: {best_acc:.2f}%) saved to {MODEL_PATH}")
    return epoch_losses, epoch_accs

def main():
    _,_ = resnet18_training_and_testing()

if __name__ == "__main__":
    main()