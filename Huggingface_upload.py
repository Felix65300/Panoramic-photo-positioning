from datasets import load_dataset

# 本地根目錄 (包含 Brightness, Mask 等子資料夾)
DATASET_PATH = "Dataset_Step2"
# 命名規範：帳號/專案名稱
REPO_NAME = "Felix96430/NSTC-Panoramic-photo-positioning"
def upload_dataset():
    print(f"🚀 開始掃描階層架構：{DATASET_PATH}...")

    dataset = load_dataset("imagefolder",data_dir=DATASET_PATH)

    print(dataset)

    print(f"☁️ 準備推送到 Hugging Face Hub ({REPO_NAME})...")
    # private=True 設定為私人，防止資料集外傳
    dataset.upload(REPO_NAME,private=True)

    print("✅ 上傳完成！")

if __name__ == "__main__":
    upload_dataset()
