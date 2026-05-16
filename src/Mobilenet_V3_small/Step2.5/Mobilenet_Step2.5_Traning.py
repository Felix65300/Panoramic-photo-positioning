import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
Mobilenet_V3_small = os.path.dirname(current_dir)
src = os.path.dirname(Mobilenet_V3_small)
Project_Root = os.path.dirname(src)

sys.path.append(Mobilenet_V3_small)
sys.path.append(src)
sys.path.append(Project_Root)

from src.data_Step2_5 import get_dataset
from src.Mobilenet_V3_small.Mobilenet_V3_small_modified_version import build_model

Learning_Rate = 1e-4
IMG_WIDTH = 512
IMG_HEIGHT = 128
BATCH_SIZE = 128
epochs = 200

MODEL_PATH = 'Step2.5_mobilenet_model.pth'


def Mobilenet_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_path = os.path.join(Project_Root, "Datasets/Dataset_Step1")

    dataset = get_dataset(root_dir=img_path, width=IMG_WIDTH, height=IMG_HEIGHT, is_train=True)

    trainloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    model = build_model(num_classes=1000)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Learning_Rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    start_epoch = 0
    best_loss = float('inf')
    epoch_losses = []

    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading weights from {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch']
            print("Weight loaded successfully.")
        except Exception as e:
            print(f"Loading failed: {e}, training from scratch.")
    else:
        print("No existing weights found. Training from scratch.")

    print("Start Training...")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, leave=True) as loop:
            for img, id_label in loop:
                img, id_label = img.to(device), id_label.to(device)

                optimizer.zero_grad()
                outputs = model(img)
                loss = criterion(outputs, id_label)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                running_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | LR: {current_lr:.8f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'best_loss': best_loss,
                'epoch': epoch + 1
            }
            torch.save(checkpoint, MODEL_PATH)
            print(f"New Best Model Saved to {MODEL_PATH}")

    return epoch_losses


def main():
    epoch_losses = Mobilenet_training()

    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.grid(True)
    plt.savefig('Mobilenet_loss_curve.png')


if __name__ == '__main__':
    main()