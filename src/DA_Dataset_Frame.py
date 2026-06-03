import glob
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as F

src = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(src)

TARGET_DIR = os.path.join(PROJECT_ROOT, 'Datasets/Dataset_Step2/')

# 🚀 1. 引入寫好的 API
from src.DA.brightness import RandomBrightness
from src.DA.color_temperature import RandomColorTemperature
from src.DA.grid_mask import GridMask
from src.DA.horizontal_roll import RandomHorizontalRoll


def generate_samples(image_path,ratios,int_id=id,is_grid_mask=False):

    original_filename = os.path.basename(image_path)
    # 1. 將檔名與副檔名切開
    # filename 會得到 "001"，ext 會得到 ".jpg"
    filename, ext = os.path.splitext(os.path.basename(original_filename))

    # 2. 強制組合成新的 PNG 檔名
    original_filename = filename + ".png"

    id = f'{int_id:03d}'
    # 🚀 2. 載入並預處理圖片 (維持 4:1，縮放至 512x128)
    with Image.open(image_path).convert('RGB') as img:
        transform_base = T.Compose([
            T.Resize((128, 512)), # PyTorch Resize 格視為 (H,W)
            T.ToTensor()
        ])
        img_tensor = transform_base(img)

    for ratio in ratios:
        api = RandomHorizontalRoll(steps=ratio,step_deg=10)
        # percent_val = int(round(ratio * 100))
        # dir_name = f"{percent_val}%"
        dir_name = f'Horizontal_Roll/{ratio*10}°'

        save_path = Path(TARGET_DIR) / dir_name / id

        save_path.mkdir(parents=True, exist_ok=True)

        save_path = save_path / original_filename

        # 執行強化 (在 no_grad 下執行，避免計算圖佔用記憶體)
        with torch.no_grad():
            # ⚠️ 注意維度：CNN 模型通常吃 (B, C, H, W)，如果 API 預設接收 Batch 維度，需加上 unsqueeze(0)
            # 如果 API 直接處理 (C, H, W)，則不需要 unsqueeze/squeeze
            # aug_tensor = api(img_tensor.unsqueeze(0)).squeeze(0)
            aug_tensor = api(img_tensor)
            # 確保數值嚴格限制在 [0.0, 1.0] 區間
            aug_tensor = torch.clamp(aug_tensor, 0.0,1.0)

        # 🚀 4. 輸出與儲存圖片
        out_pil = F.to_pil_image(aug_tensor)
        out_pil.save(save_path)

def check_api(api_class,expected_shape,**kwargs):
    print(f"\n[API 驗證] 開始測試 {api_class.__name__} ...")

    # 1. 測試實體化(建物件) (Instantiaion)
    try:
        api_instance = api_class(**kwargs)
    except Exception as e:
        return False, f"實體化失敗"

    # 2. 測試 Callable (PyTorch)
    if not callable(api_instance):
        return False, "Not Callable"

    # 3. 測試前向傳播與張量維度 ( Forward Pass & Shape Consistency)
    # 建立符合專案規格的隨機 Dummy Tensor (數值介於 0~1)
    dummy_tensor = torch.rand(3,128,512)

    try:
        with torch.no_grad():
            out_tensor = api_instance(dummy_tensor)
    except Exception as e:
        return False, f"Forward pass crashed"

    if out_tensor.shape != dummy_tensor.shape:
        return False, f"Shape Error"

    # 4. 測試數值截斷 (Clipping Validation)
    if torch.max(out_tensor) > 1.0 or torch.min(out_tensor) < 0.0:
        return False, f"Value Error"

    return True, "Approved"

# ==========================================
# ⚡ 呼叫 method ⚡
# ==========================================

if __name__ == "__main__":
    ratios = np.arange(0, 37, 1)
    source_dir = os.path.join(PROJECT_ROOT, 'Datasets/Dataset_Step1')

    for id in tqdm(range(0, 1000), desc="正在處理圖片"):
        img_dir = os.path.join(source_dir, f'{id:03d}')
        search_pattern = os.path.join(img_dir, "*.jpg")
        img_path = glob.glob(search_pattern)[0]
        generate_samples(image_path=img_path,ratios=ratios,int_id=id)