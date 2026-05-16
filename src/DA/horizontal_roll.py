import torch

class RandomHorizontalRoll:
    """
    環景圖專用增強：隨機左右平移 (Horizontal Scroll)
    由於環頸圖頭尾相連，平移不影響地理語意
    """

    def __init__(self, steps, step_deg=10):
        self.step_deg = step_deg
        self.steps = steps
        self.total_steps = 360 // step_deg

    def __call__(self, img_tensor):
        # 把 PIL 轉成 Numpy
        w = img_tensor.shape[-1]

        shift = (w // self.total_steps) * self.steps
        # axis=1 代表左右移動
        shift_tensor = torch.roll(img_tensor, shift, dims=-1)

        # 3. 轉回 PIL
        return shift_tensor