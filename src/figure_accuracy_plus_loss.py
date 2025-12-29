import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
import sys

src = os.path.dirname(os.path.abspath(__file__))
Project_Root = os.path.dirname(src)
model_dir = os.path.join(src, 'Custom_model')
fig_dir = os.path.join(Project_Root, 'Figures')
resnet18_dir = os.path.join(src, "Resnet18")
sys.path.append(model_dir)
sys.path.append(resnet18_dir)
from Custom_model.Step1_training_and_testing import model_training_and_testing
from Resnet18.resnet18_Step1_training_and_testing import resnet18_training_and_testing

# --- 1. 全局設置：讓圖片更符合論文要求 ---
plt.rcParams.update({
    'font.family': 'serif',         # 使用襯線字體
    'font.size': 12,                # 全局字體大小
    'axes.labelsize': 14,           # 軸標籤字體大小
    'axes.titlesize': 16,           # 子圖標題字體大小
    'xtick.labelsize': 12,          # X軸刻度字體大小
    'ytick.labelsize': 12,          # Y軸刻度字體大小
    'legend.fontsize': 12,          # 圖例字體大小
    'legend.frameon': True,         # 圖例加上邊框
    'legend.framealpha': 0.9,       # 圖例邊框透明度
    'lines.linewidth': 2.5,         # 線條寬度
    'axes.linewidth': 1.5,          # 座標軸線寬度
    'grid.linestyle': '--',         # 網格線樣式
    'grid.alpha': 0.6               # 網格線透明度
})

color_model = '#1f77b4' # 藍
color_resnet18 = '#d62728' # 紅

# --- 2. 準備數據 ---
epochs = np.arange(1, 201)

def main():
    model_losses, model_accs = model_training_and_testing()
    resnet18_losses,resnet18_accs = resnet18_training_and_testing()

    # ===========================
    # 2. 設定論文繪圖風格
    # ===========================
    # 使用類似 seaborn 的白底網格風格作為基礎
    plt.style.use("seaborn-v0_8-whitegrid")

    # ===========================
    # 3. 創建畫布和子圖
    # ===========================
    # 創建一個圖形框架，包含 1 列 2 行的子圖
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14,6))

    # ===========================
    # 4. 繪製左側子圖：Loss Curve 比較
    # ===========================
    # 繪製手刻模型的 Loss 曲線
    ax1.plot(epochs, model_losses, label="Model", color=color_model, linestyle='-')
    # 繪製 Resnet 18 的 Loss 曲線
    ax1.plot(epochs, resnet18_losses, label="Resnet18", color=color_resnet18, linestyle='--')

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss Value")
    ax1.set_title("Training Loss Comparison")
    ax1.legend(loc='upper right') # 圖例位置

    # ============================================
    # 5. 繪製右側子圖：Accuracy Curve 比較
    # ============================================
    # 繪製 Model A 的 Accuracy 曲線
    ax2.plot(epochs, model_accs, label="Model", color=color_model, linestyle='-')
    # 繪製 Model B 的 Accuracy 曲線
    ax2.plot(epochs, resnet18_accs, label="Resnet18", color=color_resnet18, linestyle='--')

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Comparison")
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 105)

    # ===========================
    # 6. 最後調整與存檔
    # ===========================
    # 自動調整子圖間距，避免標籤重疊
    plt.tight_layout()

    # 儲存圖片
    # dpi=300 是印刷品質的標準
    # bbox_inches='tight' 確保儲存時去除多餘白邊
    save_path = os.path.join(fig_dir, 'Loss_curve_and_Accuracy.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"圖片已儲存至: {save_path}")

    plt.show()


if __name__ == '__main__':
    main()