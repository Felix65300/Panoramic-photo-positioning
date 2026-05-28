import os
import sys
from pathlib import Path

# 1. 取得目前檔案的所在目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 取得上一層目錄 (專案的根目錄)
parent_dir = os.path.dirname(current_dir)

# 3. 將根目錄加入系統搜尋路徑
sys.path.append(parent_dir)

TARGET_DIRECTORY = Path(parent_dir) / 'Figures'

# paths = (TARGET_DIRECTORY / 'Custom_model', TARGET_DIRECTORY / 'Resnet18'
#          , TARGET_DIRECTORY / 'MobileNet-V3-Small'
#          , TARGET_DIRECTORY / 'EfficientNet-B0')


for path in paths:
    # 1. 設定目標資料夾路徑（請替換為你的資料集資料夾路徑）
    target_folder = path

    # 2. 設定你要生成的空檔名稱
    # Git 慣例使用 .gitkeep，若一定要 txt 也可以改成 "empty.txt"
    empty_filename = ".gitkeep"

    # 3. 確保目標資料夾存在
    if not target_folder.exists():
        print(f"找不到目標資料夾：{target_folder}")
    else:
        count = 0
        # 4. rglob("*") 會遞迴尋找所有層級的子資料夾
        for sub_dir in target_folder.rglob("*"):
            if sub_dir.is_dir():
                # 組合出空檔案的完整路徑
                file_path = sub_dir / empty_filename

                # exist_ok=True 代表如果檔案已經存在，就不會報錯，也不會覆蓋內容
                file_path.touch(exist_ok=True)

                print(f"已建立空檔案: {file_path}")
                count += 1

        print(f"\n處理完成！共在 {count} 個子資料夾中建立了空檔案。")