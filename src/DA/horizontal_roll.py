import numpy as np
from PIL import Image

class RandomHorizontalRoll:
    """
    環景圖專用增強：隨機左右平移 (Horizontal Scroll)
    由於環頸圖頭尾相連，平移不影響地理語意
    """
    def __call__(self, img):
        # 1. 把 PIL 轉成 Numpy
        img_np = np.array(img)

        # 2. 隨機位移
        h, w, c = img_np.shape
        shift = np.random.randint(0,w)

        # axis=1 代表左右移動
        img_np = np.roll(img_np, shift, axis=1)

        # 3. 轉回 PIL
        return Image.fromarray(img_np)