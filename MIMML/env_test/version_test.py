# ---encoding:utf-8---
# @Time : 2020.10.28
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : version_test.py

# pytorch
import torch

'''CUDA Tensors'''
print(torch.__version__)
x = torch.rand(3, 4)
y = torch.ones_like(x)
z = torch.zeros_like(x)
print(x, x.device)
print(y, y.device)
print(z, z.device)

if torch.cuda.is_available():
    # 选择GPU
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    print('device', device)
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to('cpu', torch.double))

# matplotlib, numpy
import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)


t1 = np.arange(0, 5, 0.1)
t2 = np.arange(0, 5, 0.02)

plt.figure(12)
plt.subplot(221)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'r--')

plt.subplot(222)
plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')

plt.subplot(212)
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

plt.show()
