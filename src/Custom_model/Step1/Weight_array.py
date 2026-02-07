import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
Custom_model = os.path.dirname(current_dir)
src = os.path.dirname(Custom_model)
sys.path.append(Custom_model)
sys.path.append(src)

from src.Custom_model.Convolution_Class import CNN
import torch

cnn = CNN().to("cuda")
cnn.eval()
cnn.load_state_dict(torch.load('my_model_weights.pth'))

# 獲取模型的狀態字典
state_dict = cnn.state_dict()

for name, param in state_dict.items():
    print(f"參數名稱: {name}")
    # param 是 PyTorch 的 Tensor，你可以將它轉換為 NumPy 陣列
    weights_array = param.cpu().numpy()
    print(f"  形狀: {weights_array.shape}")
    print(f"  值 (部分): {weights_array.flatten()[::]}")