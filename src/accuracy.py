import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import sys

src = os.path.dirname(os.path.abspath(__file__))
Project_Root = os.path.dirname(src)
model_dir = os.path.join(src, 'Custom_model')
resnet18_dir = os.path.join(src, "Resnet18")
sys.path.append(model_dir)
sys.path.append(resnet18_dir)
from Custom_model.Inference_1000 import model_testing
from Resnet18.inference_resnet18 import resnet18_testing

# --- 1. 全局設置：讓圖片更符合論文要求
# 設置全局字體大小，確保縮圖後文字依然清晰
plt.rcParams.update({'font.size': 14})
# 設置線條默認粗細
plt.rcParams['lines.linewidth'] = 2.5

# --- 2. 準備數據
figures = 1000


def model():
    model_accuracy,model_correct,model_total = model_testing()
    model_final_accuracy = 100 * model_correct / model_total
    print(f"{'=' * 50}")
    print(f"📊 手刻模型最終測試結果")
    print(f"  - 測試張數: {model_total} 張")
    print(f"  - 答對張數: {model_correct} 張")
    print(f"  - 答錯張數: {model_total - model_correct} 張")
    print(f"🏆 總正確率 (Accuracy): {model_final_accuracy:.2f}%")
    print(f"{'=' * 50}")
    return model_accuracy

def resnet18():
    resnet18_accuracy,resnet18_correct,resnet18_total = resnet18_testing()
    resnet18_final_accuracy = 100 * resnet18_correct / resnet18_total
    print(f"{'=' * 50}")
    print(f"📊 Resnet 最終測試結果")
    print(f"  - 測試張數: {resnet18_total} 張")
    print(f"  - 答對張數: {resnet18_correct} 張")
    print(f"  - 答錯張數: {resnet18_total - resnet18_correct} 張")
    print(f"🏆 總正確率 (Accuracy): {resnet18_final_accuracy:.2f}%")
    print(f"{'=' * 50}")
    return resnet18_accuracy
def main():
    model_accuracy = model()
    resnet18_accuracy = resnet18()

    # --- 2. 準備數據 ---
    batches = np.arange(1,1001)

    # --- 3. 開始繪圖 ---
    # 創建一個畫布，figsize=(10,7) 是一個適合論文單欄或跨欄的比例
    plt.figure(figsize=(10,7))

    # === 繪製手刻模型 (藍色) ===
    plt.plot(batches, model_accuracy, label="Model Accuracy", color='#1f77b4',linestyle='-')

    # === 繪製Resnet模型 (紅色) ===
    plt.plot(batches, resnet18_accuracy, label="Resnet 18", color='#d62728',linestyle='-')

    # --- 4. 添加圖表細節 ---
    # 標題與軸標籤
    plt.title("Comparison of Accuracy (Our model vs. Resnet 18)", fontweight='bold', pad=15)
    plt.xlabel("Number of Batches", labelpad=10)
    plt.ylabel("Accuracy", labelpad=10)

    # 添加網格線，使用灰色虛線，增加可讀性但不搶戲
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # 設置X、Y軸上下限
    plt.ylim(bottom=0, top=200)
    plt.xlim(left=1, right=1000)

    # 添加圖例
    # frameon=True 加上邊框，shadow=True 加上陰影
    plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=12)

    # 自動調整佈局
    plt.tight_layout()

    # --- 5. 儲存圖片 ---
    save_path = Path(Project_Root) / 'Figures' / 'Accuracy.png'
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    print("圖片生成成功")
if __name__ == '__main__':
    main()