import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

from torchvision.models import resnet18

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
model_dir = os.path.join(current_dir, 'Custom_model')
resnet18_dir = os.path.join(parent_dir, "resnet18")
Project_Root = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(Project_Root)
sys.path.append(model_dir)
from Custom_model.Inference_1000 import testing
from Resnet18.resnet18_revised_version import get_pano_model

# --- 1. 全局設置：讓圖片更符合論文要求
# 設置全局字體大小，確保縮圖後文字依然清晰
plt.rcParams.update({'font.size': 14})
# 設置線條默認粗細
plt.rcParams['lines.linewidth'] = 2.5

# --- 2. 準備數據
figures = 1000


def main():
    model_accuracy,model_correct,model_total = testing()
    model_final_accuracy = 100 * model_correct / model_total
    print(f"{'=' * 50}")
    print(f"📊 最終測試結果")
    print(f"  - 測試張數: {model_total} 張")
    print(f"  - 答對張數: {model_correct} 張")
    print(f"  - 答錯張數: {model_total - model_correct} 張")
    print(f"🏆 總正確率 (Accuracy): {model_final_accuracy:.2f}%")
    print(f"{'=' * 50}")
    print()

if __name__ == '__main__':
    main()