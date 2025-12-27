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
from Custom_model.Step1_training_and_testing import model_training_and_testing
from Resnet18.resnet18_Step1_training_and_testing import resnet18_training_and_testing

# --- 1. 全局設置：讓圖片更符合論文要求 ---
# 設置全局字體大小，確保縮圖後文字依然清晰
plt.rcParams.update({'font.size': 14})
# 設置線條默認粗細
plt.rcParams['lines.linewidth'] = 2.5

# --- 2. 準備數據 ---
figures = 1000

def main():
    model_losses, model_accs = model_training_and_testing()
    resnet18_losses,resnet18_accs = resnet18_training_and_testing()

if __name__ == '__main__':
    main()