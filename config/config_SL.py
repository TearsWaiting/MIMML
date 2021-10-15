import argparse


def get_config():
    parse = argparse.ArgumentParser(description='common supervised learning config')

    # 项目配置参数
    # parse.add_argument('-learn-name', type=str, default='SL_train_00', help='本次训练的名称')
    parse.add_argument('-save-best', type=bool, default=True, help='当得到更好的准确度是否要保存')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=0)
    parse.add_argument('-seed', type=int, default=50)

    # 路径参数
    parse.add_argument('-path-token2index', type=str, default='../data/meta_data/residue2idx.pkl', help='保存字典的位置')
    parse.add_argument('-path-train-data', type=str, default='../data/task_data/IL-6/Train.tsv', help='训练数据的位置')
    parse.add_argument('-path-test-data', type=str, default='../data/task_data/IL-6/Validate.tsv', help='测试数据的位置')
    # parse.add_argument('-path-dataset', type=str, default='../data/task_data/BPD-36', help='多分类训练数据的位置')
    parse.add_argument('-path-dataset', type=str, default=None, help='多分类训练数据的位置')
    parse.add_argument('-path-params', type=str, default=None, help='模型参数路径')
    parse.add_argument('-path-save', type=str, default='../result/', help='保存字典的位置')
    # parse.add_argument('-model-save-name', type=str, default='TE', help='保存模型的命名')
    parse.add_argument('-model-save-name', type=str, default='CNN', help='保存模型的命名')
    parse.add_argument('-save-figure-type', type=str, default='png', help='保存图片的文件类型')

    # 数据参数
    parse.add_argument('-num-class', type=int, default=2, help='类别数量')
    # parse.add_argument('-num-class', type=int, default=28, help='类别数量')
    parse.add_argument('-max-len', type=int, default=207, help='max length of input sequences')
    parse.add_argument('-dataset', type=str, default='None', help='数据集名称')

    # 框架参数
    parse.add_argument('-mode', type=str, default='train-test', help='训练模式')
    # parse.add_argument('-mode', type=str, default='cross validation', help='训练模式')
    # parse.add_argument('-k-fold', type=int, default=5, help='k折交叉验证')
    parse.add_argument('-interval-log', type=int, default=20, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval-valid', type=int, default=1, help='经过多少epoch对交叉验证集进行测试')
    parse.add_argument('-interval-test', type=int, default=1, help='经过多少epoch对测试集进行测试')
    # parse.add_argument('-metric', type=str, default='MCC', help='评估指标名称')
    parse.add_argument('-metric', type=str, default='ACC', help='评估指标名称')
    # parse.add_argument('-threshold', type=float, default=0.45, help='指标率阈值')
    parse.add_argument('-threshold', type=float, default=0.40, help='指标率阈值')

    # 训练参数
    parse.add_argument('-model', type=str, default='TextCNN', help='模型名称')
    # parse.add_argument('-model', type=str, default='TextCNN_finetune', help='模型名称')
    # parse.add_argument('-model', type=str, default='Transformer Encoder', help='模型名称')
    parse.add_argument('-optimizer', type=str, default='AdamW', help='优化器名称')
    parse.add_argument('-loss-func', type=str, default='FL', help='损失函数名称, CE/FL')
    # parse.add_argument('-lr', type=float, default=0.0001, help='学习率')
    # parse.add_argument('-lr', type=float, default=0.0003, help='学习率')
    parse.add_argument('-lr', type=float, default=0.0005, help='学习率')
    parse.add_argument('-reg', type=float, default=0.0025, help='正则化lambda')
    # parse.add_argument('-epoch', type=int, default=50, help='迭代次数')
    parse.add_argument('-epoch', type=int, default=80, help='迭代次数')
    parse.add_argument('-batch-size', type=int, default=32, help='一个batch中有多少个sample')

    # Focal Loss参数
    parse.add_argument('-gamma', type=float, default=2, help='gamma in Focal Loss')
    parse.add_argument('-alpha', type=float, default=None, help='alpha in Focal Loss')
    # parse.add_argument('-alpha', type=float, default=0.1, help='alpha in Focal Loss')

    # Transformer Encoder 模型参数
    # # parse.add_argument('-num-layer', type=int, default=3, help='Transformer的Encoder模块的堆叠层数')
    # parse.add_argument('-num-layer', type=int, default=6, help='Transformer的Encoder模块的堆叠层数')
    # parse.add_argument('-dropout', type=float, default=0.5, help='dropout率')
    # parse.add_argument('-static', type=bool, default=False, help='嵌入是否冻结')
    # parse.add_argument('-num-head', type=int, default=8, help='多头注意力机制的头数')
    # parse.add_argument('-dim-embedding', type=int, default=128, help='词（残基）向量的嵌入维度')
    # parse.add_argument('-dim-feedforward', type=int, default=32, help='词（残基）向量的嵌入维度')
    # parse.add_argument('-dim-k', type=int, default=32, help='k/q向量的嵌入维度')
    # parse.add_argument('-dim-v', type=int, default=32, help='v向量的嵌入维度')

    # TextCNN 模型参数
    parse.add_argument('-dim-embedding', type=int, default=128, help='词（残基）向量的嵌入维度')
    parse.add_argument('-dropout', type=float, default=0.5, help='dropout率')
    parse.add_argument('-static', type=bool, default=False, help='嵌入是否冻结')
    parse.add_argument('-num-filter', type=int, default=128, help='卷积核的数量')
    parse.add_argument('-filter-sizes', type=str, default='1,2,4,8,16,24,32,64', help='卷积核的尺寸')
    parse.add_argument('-dim-cnn-out', type=int, default=128, help='CNN模型的输出维度')
    # parse.add_argument('-output-extend', type=str, default='pretrain', help='CNN后是否再接一层')
    parse.add_argument('-output-extend', type=str, default='finetune', help='CNN后是否再接一层')

    config = parse.parse_args()
    return config
