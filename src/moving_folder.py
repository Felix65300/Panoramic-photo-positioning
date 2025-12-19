import os
import sys

# 1. 取得目前檔案的所在目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 取得上一層目錄 (專案的根目錄)
parent_dir = os.path.dirname(current_dir)

# 3. 將根目錄加入系統搜尋路徑
sys.path.append(parent_dir)

# 程式開始
import shutil

TARGET_DIRECTORY = os.path.join(parent_dir,"_gkcN1hzqm1RFcsvpk5Xmg")
print(TARGET_DIRECTORY)

# 1. 取得該資料夾內的所有檔案
# 這裡會過濾掉子資料夾，只抓取檔案
files = [f for f in os.listdir(TARGET_DIRECTORY) if os.path.isfile((os.path.join(TARGET_DIRECTORY,f)))]

# 設定允許的圖片副檔名，避免移動到程式碼本身或系統檔
valid_extensions = ('.jpg')

# 過濾出圖片
images = [f for f in files if f.lower().endswith(valid_extensions)]

print(f"共發現{len(images)} 張圖片，準備開始處理...")

count = 0
id = 0
for image_name in images:
    # 3. 定義來源路徑
    src_path = os.path.join(TARGET_DIRECTORY,image_name)

    # 4. 決定新資料夾名稱
    folder_name = f'{id:03d}'
    id += 1

    # 5. 建立新資料夾路徑
    new_folder_path = os.path.join(TARGET_DIRECTORY, folder_name)

    # 建立資料夾 (exist_ok = Ture 表示如果資料夾已存在也不會報錯)
    os.makedirs(new_folder_path, exist_ok=True)

    # 6. 移動圖片
    dst_path = os.path.join(new_folder_path, image_name)
    shutil.move(src_path, dst_path)

    count += 1
    # 每處理 100 張印出一次進度
    if count % 100 == 0:
        print(f"已處理 {count} 張圖片...")

print(f"子資料夾建立與圖片移動已全數完成")
