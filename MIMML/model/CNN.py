# ---encoding:utf-8---
# @Time : 2021.02.25
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : CNN.py

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),  # 维度变换(1,28,28) --> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 维度变换(16,28,28) --> (16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),  # 维度变换(16,14,14) --> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 维度变换(32,14,14) --> (32,7,7)
        )
        self.output = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # print(x) # [batch, channel, height, width]
        out = self.conv1(x)  # 维度变换(Batch,1,28,28) --> (Batch,16,14,14)
        out = self.conv2(out)  # 维度变换(Batch,16,14,14) --> (Batch,32,7,7)
        out = out.view(out.size(0), -1)  # 维度变换(Batch,32,14,14) --> (Batch,32*14*14)||将其展平
        out = self.output(out)
        return out


if __name__ == '__main__':
    cnn = CNN()
    print(cnn)
