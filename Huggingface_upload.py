import os

from huggingface_hub import HfApi

# 🚀 1. 初始化 API
api = HfApi()
repo_name = "Felix96430/Panoramic-photo-positioning"

print("啟動 API 穩健上傳模式")

# 🚀 3. 執行資料夾同步 (支援斷點續傳)
api.upload_large_folder(
    folder_path="Datasets/Dataset_Step2",
    repo_id=repo_name,
    repo_type="dataset"
)


print("✅ 所有圖片上傳完畢！")