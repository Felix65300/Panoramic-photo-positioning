from huggingface_hub import HfApi
from datasets import load_dataset

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

def verify_upload():

        # 直接讀取你的新組織專案
        dataset = load_dataset("Panoramic-photo-positioning/Panoramic-photo-positioning-Step2")

        print("-" * 30)
        # dataset['train'] 是預設的 split
        print(f"✅ 解析成功！PyTorch DataLoader 可讀取的總筆數為：{dataset['train'].num_rows}")
if __name__ == "__main__":
    upload_to_huggingface()
    # verify_upload()