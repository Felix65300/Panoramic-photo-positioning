import numpy as np
from torchvision import transforms, datasets
from PIL import Image

# 定義一個客製化的 Transform：環景圖隨機左右位移
class RandomHorizontalRoll:
    def __call__(self, img):
        # 1. 把 PIL 轉成 Numpy (因為 np.roll 比較好寫)
        img_np = np.array(img)

        # 2. 隨機位移
        h,w,c = img_np.shape
        shift = np.random.randint(0,w)
        # axis = 1 = > 左右移動，axix = 0 = > 上下移動
        img_np = np.roll(img_np,shift,axis=1)

        # 3. 轉回 PIL
        return Image.fromarray(img_np)

# 提供一個函式讓外部取得標準的 Transform
def get_transforms(img_width, img_height, is_train=True):
    transform_list = []

    # 1. Resize
    transform_list.append(transforms.Resize((img_height, img_width)))

    # 2. 訓練時加入隨機位移
    if is_train:
        transform_list.append(RandomHorizontalRoll())

    # 3. 轉 Tensor (自動做 / 255.0 和 HWC->CHW)
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)
# ---------------------------------------------------
# 3. 【關鍵】 取得 Dataset 的包裝函式
# 這裡把 ImageFolder 藏起來，外部只要呼叫這個函式就好
# ---------------------------------------------------
def get_dataset(root_dir, width, height, is_train=True):
    # 1. 取得對應的 Transform
    my_transform = get_transforms(width, height, is_train)

    # 2. 建立 ImageFolder
    # ImageFolder 會自動掃描 root_dir 下的所有子資料夾
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transform)
    return dataset

