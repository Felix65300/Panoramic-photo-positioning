import torch
import torch.nn as nn

class CNN(nn.Module):
  def __init__(self, num_classes = 1000):
    super(CNN, self).__init__()

    # 定義卷積區塊
    def conv_block(in_channels, out_channels):
      return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3,padding = 1, padding_mode= 'circular'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
      )

    # 5 層卷積，通道數逐漸大
    self.layer1 = conv_block(3,32)
    self.layer2 = conv_block(32,64)
    self.layer3 = conv_block(64,128)
    self.layer4 = conv_block(128,256)
    self.layer5 = conv_block(256,512)

    # ----------------------
    # 加入全域平均池化 (GAP)
    # ----------------------
    # 不管前面圖片剩多大 (例如 16 x 4)，強制壓縮成 1 x 1
    self.gap = nn.AdaptiveAvgPool2d((1,1))

    # ----------------------
    # 瘦身後的分類器
    # ----------------------
    # 因為經過 GAP，輸入特徵固定就是 512 (最後一層的通道數)
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, num_classes)
    )

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)

    x = self.gap(x) # 通過 GAP
    x = self.fc(x)  # 通過分類器
    return x