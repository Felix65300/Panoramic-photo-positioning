from huggingface_hub import HfApi

def upload_to_huggingface():
    # 🚀 1. 初始化 API
    api = HfApi()
    repo_name = "Panoramic-photo-positioning/Panoramic-photo-positioning-Step2"

    print(f"啟動 API 穩健上傳模式，REPO NAME: {repo_name}")

    api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)

    # 🚀 2. 執行資料夾同步 (支援斷點續傳)
    api.upload_large_folder(
        folder_path="Datasets/Dataset_Step2",
        repo_id=repo_name,
        repo_type="dataset"
    )


    print("✅ 所有圖片上傳完畢！")

if __name__ == "__main__":
    upload_to_huggingface()