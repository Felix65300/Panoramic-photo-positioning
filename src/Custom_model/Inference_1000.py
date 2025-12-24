import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 路徑設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root)

from src.data import get_dataset
from Convolution_Class import CNN
from matplotlib import pyplot as plt
# ---------------------------------------------------------
# 參數設定
# ---------------------------------------------------------
MODEL_PATH = 'pano_cnn_model.pth'
IMG_DIR = os.path.join(project_root, 'Dataset_Step1')

IMG_WIDTH = 512
IMG_HEIGHT = 128
BATCH_SIZE = 32
DEVICE = torch.device("cuda")

def testing():
    # 1. 準備 Dataset (讀取全部圖片)
    # is_train=False 代表不做隨機位移，測試原始圖片
    test_dataset = get_dataset(IMG_DIR, IMG_WIDTH, IMG_HEIGHT, is_train=False)

    # shuffle=True: 確保 1000 張圖片順序是被打亂的
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 載入模型 (包含 Checkpoint)
    model = CNN().to(DEVICE)
    if os.path.isfile(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_loss = checkpoint.get('best_loss', 'N/A')
            print(f"--> 模型載入成功 (Training Best Loss: {best_loss})")

    # 3. 開始全數測試
    model.eval()
    correct = 0
    total = 0

    print(f"--> 開始測試 1000 張圖片")
    print(f"{'=' * 50}")

    epoch_accuracy = []
    # 使用 tqdm 顯示進度條
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit='batch'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_accuracy.append(100 * correct / total)

    return epoch_accuracy,correct,total

def main():
    epoch_accuracy,correct,total = testing()
    # 4. 結算成績
    accuracy = 100 * correct / total

    print(f"{'='*50}")
    print(f"📊 最終測試結果")
    print(f"   - 測試總數: {total} 張")
    print(f"   - 答對張數: {correct} 張")
    print(f"   - 答錯張數: {total - correct} 張")
    print(f"🏆 總正確率 (Accuracy): {accuracy:.2f}%")
    print(f"{'='*50}")

    # 5. 存圖表
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_accuracy, label='Training Loss')
    plt.grid(True)
    plt.savefig('Accuracy.png')


if __name__ == '__main__':
    main()