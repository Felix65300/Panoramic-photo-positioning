import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

src = os.path.dirname(__file__)
Project_Root = os.path.dirname(src)
sys.path.append(Project_Root)
sys.path.append(src)
Custom_model = os.path.join(src,'Custom_model')
Resnet18 = os.path.join(src,'Resnet18')
sys.path.append(Resnet18)
sys.path.append(Custom_model)

from Custom_model.Training import model_training
from Resnet18.resnet18_train import resnet18_training

# --- 1. 全局設置 ---
# 設置全局字體大小
plt.rcParams.update({'font.size':14})
# 設置線條默認粗細
plt.rcParams['lines.linewidth'] = 2.5

# --- 2. 準備數據 ---
epoches = np.arange(1,101)

def main():
    model_loss = model_training()
    resnet18_loss = resnet18_training()

    # --- 3. 開始繪圖 ---
    plt.figure(figsize=(10,7))

    # === 手刻模型(藍色) ===
    plt.plot(epoches, model_loss, label='Model Loss', color='#1f77b4',linestyle='-')

    # === Resnet18(紅色)
    plt.plot(epoches,resnet18_loss, label='Resnet18 Loss', color="#d62728", linestyle='-')

    # --- 4. 添加圖表細節 ---
    plt.title("Comparison of Loss Curves (Model vs. Resnet18)", fontweight='bold', pad=15)
    plt.xlabel("Epochs", labelpad=10)
    plt.ylabel("Loss (Cross Entropy)", labelpad=10)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpah=0.5)

    # 設置 X、Y 軸上下限
    plt.ylim(bottom=0)
    plt.xlim(left=1, right=1000)

    # 添加圖例
    plt.legend(loc='upper right', frameon=True, shadow=True, fonsize=12)

    plt.tight_layout()

    # --- 5. 儲存圖片 ---
    save_path = Path(Project_Root) / 'Figures' / 'Loss_curve.png'
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()