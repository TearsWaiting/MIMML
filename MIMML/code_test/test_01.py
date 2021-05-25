# ---encoding:utf-8---
# @Time : 2021.03.17
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : test_01.py

import numpy as np
import torch

from random import shuffle


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a1 = a.unsqueeze(1)
    a2 = a1.expand(n, m, -1)
    b1 = b.unsqueeze(0).expand(n, m, -1)
    b2 = b1.expand(n, m, -1)
    logits1 = -((a1 - b1) ** 2)
    logits = logits1.sum(dim=2)
    return logits


def get_entropy(probs):
    ent = - (probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
    return ent


if __name__ == '__main__':
    # a = torch.rand([15, 32])
    # b = torch.rand([5, 32])
    # pairwise_distances_logits(a, b)

    # a = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    # a = a.reshape(3, 2, -1).float()
    # print(a)
    # a = a.mean(dim=1)
    # print(a)

    # a = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]).float()
    # a = a.softmax(1)
    # print(a)

    '''
    测试成对距离的计算和信息熵的计算
    '''

    a = torch.tensor([[0, 1, 2], [2, 3, 4]]).float()
    b = torch.tensor([[1, 2, 1], [1, 2, 1]]).float()
    logb = torch.log(b)
    ab = a * b
    print('logb', logb)
    print('ab', ab)

    probs_a = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]).float()
    get_entropy(probs_a)
