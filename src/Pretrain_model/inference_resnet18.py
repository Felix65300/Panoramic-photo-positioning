import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm 

# ---------------------------------------------------------
# 環境與路徑設定 (與 Training 檔保持一致)
# ---------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
Project_Root = os.path.dirname(parent_dir)

sys.path.append(parent_dir)
sys.path.append(Project_Root)

# Import 自定義模組
from src.data import get_dataset
from resnet18_revised_version import get_pano_model

# ---------------------------------------------------------
# 參數設定
# ---------------------------------------------------------
IMG_WIDTH = 512
IMG_HEIGHT = 128
BATCH_SIZE = 32  # 推論時 Batch Size 可以隨意設，不影響結果
MODEL_PATH = os.path.join(current_dir, 'resnet18_pano_1000classes_optimized.pth')

def main():
    # 1. 設定裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 準備資料路徑
    img_path = os.path.join(Project_Root, "_gkcN1hzqm1RFcsvpk5Xmg")
    
    # 3. 載入 Dataset
    # 注意：這裡 is_train=False，代表不做隨機位移，只做 Resize 和 Tensor 轉換
    # 這樣才能測試圖片「標準狀態」下的準確率
    print("Loading dataset...")
    dataset = get_dataset(root_dir=img_path, width=IMG_WIDTH, height=IMG_HEIGHT, is_train=False)
    
    # 4. 建立 DataLoader
    # 這裡設定 shuffle=True，滿足你「打亂順序」的要求
    test_loader = DataLoader(
        dataset=dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    print(f"Total images: {len(dataset)}")

    # 5. 載入模型架構
    model = get_pano_model(num_classes=1000, pretrained=False)
    model = model.to(device)

    # 6. 載入權重
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading weights from: {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint)
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # 7. 開始推論 (Inference)
    model.eval()  # 【關鍵】切換到評估模式 (固定 BN, 關閉 Dropout)
    
    correct = 0
    total = 0
    
    print("Start Inference...")
    
    # torch.no_grad() 讓 PyTorch 知道不用算梯度，省記憶體且加速
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", ncols=100) as loop:
            for imgs, labels in loop:
                imgs, labels = imgs.to(device), labels.to(device)

                # 前向傳播
                outputs = model(imgs)
                
                # 取得預測結果 (機率最大的那個類別 index)
                _, predicted = torch.max(outputs, 1)
                
                # 統計
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 即時更新進度條上的準確率
                current_acc = 100 * correct / total
                loop.set_postfix(acc=f"{current_acc:.2f}%")

    # 8. 輸出最終結果
    final_acc = 100 * correct / total
    print("-" * 30)
    print(f"Final Accuracy: {final_acc:.2f}%")
    print(f"Correct: {correct} / {total}")
    print("-" * 30)

if __name__ == '__main__':
    main()