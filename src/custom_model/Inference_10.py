import os
import sys
import torch
from torch.utils.data import DataLoader

# --- 路徑設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dit = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dit)
sys.path.append(parent_dit)
sys.path.append(project_root)

from src.data import get_dataset
from Convolution_Class import CNN

# 參數
MODEL_PATH = 'pano_cnn_model.pth'
IMG_DIR = os.path.join(project_root,'Dataset_Step1')
IMG_WIDTH = 512
IMG_HEIGHT = 128
BATCH_SIZE = 10
DEVICE = torch.device("cuda")

def main():
    # 1. 準備 Dataset (is_train=False)
    # 這樣就不用管內部是用 ImageFolder 還是什麼，只要告訴它路徑就好
    test_dataset = get_dataset(IMG_DIR, IMG_WIDTH, IMG_HEIGHT, is_train=False)

    # 建立 ID 轉 檔名的字典 (0 -> '001')
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 載入模型
    model = CNN(num_classes=len(test_dataset.classes)).to(DEVICE)
    # A. 讀取檔案
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # B.判斷格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint.get('best_loss', '未知')
    model.eval()

    # 3. 測試
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    print(f'\n{'='*40}')
    print(f'{'真實':<10} | {'預測':<10} | {'結果'}')
    print(f'{'='*40}')

    correct = 0
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs.data, 1)

        for i in range(BATCH_SIZE):
            true_name = idx_to_class[labels[i].item()]
            pred_name = idx_to_class[predictions[i].item()]

            is_correct = (true_name == pred_name)
            if is_correct: correct += 1

            print(f"{true_name:<10} | {pred_name:<10} | {'☑️' if is_correct else '✖️'}")

    print(f"{'='*40}")
    print(f"準確率: {correct/BATCH_SIZE * 100:.1F}%")

if __name__ == '__main__':
    main()



