# ---encoding:utf-8---
# @Time : 2021.03.17
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : test_tqdm.py

import time
from tqdm import tqdm
from tqdm import trange
from colorama import Fore

# for i in tqdm(range(1000)):
#     # print(i)
#     pass

# for char in tqdm(["a", "b", "c", "d"]):
#     print("Start : %s" % time.ctime())
#     time.sleep(5)
#     print("End : %s" % time.ctime())
#     pass

for i in trange(100, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.LIGHTMAGENTA_EX, Fore.LIGHTMAGENTA_EX)):
    time.sleep(0.1)
    pass

pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    # 设置进度条左边显示的信息
    pbar.set_description("Processing %s" % char)
    # 设置进度条右边显示的信息
    accuracy = 0.8
    pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy))

print(3e-3)
