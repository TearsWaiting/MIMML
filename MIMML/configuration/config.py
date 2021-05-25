# ---encoding:utf-8---
# @Time : 2021.02.25
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : config.py


import argparse


def get_train_config():
    parse = argparse.ArgumentParser(description='common supervised learning config')

    # 项目配置参数
    parse.add_argument('-learn-name', type=str, default='train_00', help='本次训练的名称')
    parse.add_argument('-path-meta-data', type=str, default='../data/meta_data/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=False, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0.90, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=0)

    # 训练参数
    # parse.add_argument('-lr', type=float, default=0.0001, help='学习率')
    # parse.add_argument('-lr', type=float, default=0.0003, help='学习率')
    parse.add_argument('-lr', type=float, default=0.0005, help='学习率')
    # parse.add_argument('-lr', type=float, default=0.00007, help='学习率')
    # parse.add_argument('-lr', type=float, default=0.00005, help='学习率')
    parse.add_argument('-reg', type=float, default=0.0025, help='正则化lambda')
    parse.add_argument('-batch-size', type=int, default=32, help='一个batch中有多少个sample')
    parse.add_argument('-epoch', type=int, default=50, help='迭代次数')
    parse.add_argument('-k-fold', type=int, default=-1, help='k折交叉验证,-1代表只使用train-test方式')
    parse.add_argument('-num-class', type=int, default=2, help='类别数量')
    parse.add_argument('-train-way', type=int, default=2, help='类别数量')
    parse.add_argument('-interval-log', type=int, default=20, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval-valid', type=int, default=1, help='经过多少epoch对交叉验证集进行测试')
    parse.add_argument('-interval-test', type=int, default=1, help='经过多少epoch对测试集进行测试')

    # 模型参数
    # 通用
    parse.add_argument('-dim-embedding', type=int, default=32, help='词（残基）向量的嵌入维度')
    parse.add_argument('-num-layer', type=int, default=2, help='Transformer的Encoder模块的堆叠层数')
    parse.add_argument('-dropout', type=float, default=0.4, help='dropout率')
    parse.add_argument('-static', type=bool, default=False, help='嵌入是否冻结')

    # BERT
    # parse.add_argument('-max-len', type=int, default=207 + 2, help='max length of input sequences')
    parse.add_argument('-num-head', type=int, default=8, help='多头注意力机制的头数')
    parse.add_argument('-dim-feedforward', type=int, default=32, help='词（残基）向量的嵌入维度')
    parse.add_argument('-dim-k', type=int, default=32, help='k/q向量的嵌入维度')
    parse.add_argument('-dim-v', type=int, default=32, help='v向量的嵌入维度')

    # TextCNN
    parse.add_argument('-num-filter', type=int, default=32, help='卷积核的数量')
    parse.add_argument('-filter-sizes', type=str, default='1,2,4,8,16,24,32,50', help='卷积核的尺寸')
    # parse.add_argument('-filter-sizes', type=str, default='1,2,4,8,16,24,32,41', help='卷积核的尺寸')

    # TextRNN
    parse.add_argument('-hidden-size', type=int, default=32, help='隐藏层大小')
    parse.add_argument('-bidirectional', type=bool, default=True, help='是否双向')

    config = parse.parse_args()
    return config
