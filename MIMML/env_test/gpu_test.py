# ---encoding:utf-8---
# @Time : 2020.10.28
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : gpu_test.py

import torch
import pynvml  # TODO 记得需要先安装该包才能运行

print('GPU is_available', torch.cuda.is_available())

pynvml.nvmlInit()
gpu_num = pynvml.nvmlDeviceGetCount()
print('gpu num:', gpu_num)  # 显示有几块GPU

for i in range(gpu_num):
    print('-' * 50, 'gpu[{}]'.format(str(i)), '-' * 50)
    gpu = pynvml.nvmlDeviceGetHandleByIndex(i)
    print('gpu object:', gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu)
    print('total memory:', meminfo.total / 1024 ** 3, 'GB')  # 第i块显卡总的显存大小
    print('using memory:', meminfo.used / 1024 ** 3, 'GB')  # 这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    print('remaining memory:', meminfo.free / 1024 ** 3, 'GB')  # 第二块显卡剩余显存大小
