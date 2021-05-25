# ---encoding:utf-8---
# @Time : 2020.12.24
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : util_file.py

import os


# 读取txt文本的全部内容
def read_txt(txt_filename, skip_first=False):
    with open(txt_filename, 'r') as file:
        if skip_first:
            next(file)
        content = file.read()
    return content


# 将两个列表以表格的形式化写入tsv，格式是’index    label    text‘
def write_tsv_format_data(tsv_filename, labels, sequences):
    if len(labels) == len(sequences):
        with open(tsv_filename, 'w') as file:
            file.write('index\tlabel\ttext\n')
            for i in range(len(labels)):
                file.write('{}\t{}\t{}\n'.format(i, labels[i], sequences[i]))
        return True
    return False


# 读取tsv文件
def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))

    return sequences, labels


# 合并写入tsv文件
def merge_tsv(tsv_filename, content1, content2):
    with open(tsv_filename, 'w') as file:
        file.write('index\tlabel\ttext\n')
        file.write(content1)
        file.write(content2)
    return True


# 通用tsv文件写入工具
def write_tsv_common(tsv_filename, head, list):
    num_line = len(list[0])
    with open(tsv_filename, 'w') as file:
        file.write(head)
        for i in range(num_line):
            for j in range(len(list)):
                file.write('{}'.format(list[j][i]))
                if j + 1 < len(list):
                    file.write('\t')
            file.write('\n')


# 通用tsv文件读取工具
def load_tsv_common(tsv_filename, skip_head=True):
    with open(tsv_filename, 'r') as file:
        head = file.readline()

    num_list = len(head.split('\t'))
    list = [[] for i in range(num_list)]

    with open(tsv_filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            content = line.split('\t')
            for i in range(num_list):
                list[i].append(content[i])
    return list


# 遍历给定文件夹
def walkFile(filename):
    res = []
    for root, dirs, files in os.walk(filename):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            file = os.path.join(root, f)
            res.append(file)
            print('f', file)

        # 遍历所有的文件夹
        # for d in dirs:
        #     print('d', os.path.join(root, d))
    return res
