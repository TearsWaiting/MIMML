# ---encoding:utf-8---
# @Time : 2021.03.17
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : useful_funtion.py


import os
import time
from tqdm import tqdm
from tqdm import trange
from colorama import Fore


# 执行绝对路径的python文件
def invoking_python_scipt(filename):
    os.system('python ' + filename)


# 打印进度条
def processing_bar():
    for i in trange(50, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.LIGHTMAGENTA_EX, Fore.LIGHTMAGENTA_EX)):
        time.sleep(0.1)
    pass

    pbar = tqdm(["a", "b", "c", "d"])
    for char in pbar:
        # 设置进度条左边显示的信息
        pbar.set_description("Processing %s" % char)
        # 设置进度条右边显示的信息
        accuracy = 0.8
        pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy))


if __name__ == '__main__':
    filename = 'run_test.py'
    invoking_python_scipt(filename)
    processing_bar()
