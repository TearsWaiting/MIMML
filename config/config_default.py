import argparse


def get_config():
    parse = argparse.ArgumentParser(description='default config')
    parse.add_argument('-task-type-run', type=str, default=None, help='本次训练需要运行的脚本类型')
    parse.add_argument('-path-config', type=str, default=None, help='本次训练需要加载的配置路径')

    # 项目配置参数
    # Common
    parse.add_argument('-learn-name', type=str, default=None, help='本次训练名称')
    parse.add_argument('-process-name', type=str, default=None, help='Pycharm进程的名称')
    parse.add_argument('-save-best', type=bool, default=None, help='当得到更好的准确度是否要保存')
    parse.add_argument('-cuda', type=bool, default=None)
    parse.add_argument('-device', type=int, default=None)
    parse.add_argument('-seed', type=int, default=None)
    parse.add_argument('-num_workers', type=int, default=None)

    # 路径参数
    # Supervised Learning
    parse.add_argument('-path-train-data', type=str, default=None, help='训练数据的位置')
    parse.add_argument('-path-test-data', type=str, default=None, help='测试数据的位置')

    # Meta Learning
    parse.add_argument('-path-token2index', type=str, default=None, help='保存字典的位置')
    parse.add_argument('-path-meta-dataset', type=str, default=None, help='元学习数据的位置')

    # Common
    parse.add_argument('-path-params', type=str, default=None, help='模型参数路径')
    parse.add_argument('-path-save', type=str, default=None, help='保存字典的位置')
    parse.add_argument('-model-save-name', type=str, default=None, help='保存模型的命名')
    parse.add_argument('-save-figure-type', type=str, default=None, help='保存图片的文件类型')

    # 数据参数
    # Supervised Learning
    parse.add_argument('-num-class', type=int, default=None, help='类别数量')

    # Meta Learning
    parse.add_argument('-dataset', type=str, default=None, help='数据集')

    # Common
    parse.add_argument('-max-len', type=int, default=None, help='max length of input sequences')

    # 框架参数
    # Supervised Learning
    parse.add_argument('-interval-log', type=int, default=None, help='经过多少batch记录一次训练状态')
    parse.add_argument('-interval-valid', type=int, default=None, help='经过多少epoch对交叉验证集进行测试')
    parse.add_argument('-interval-test', type=int, default=None, help='经过多少epoch对测试集进行测试')

    # Meta Learning
    parse.add_argument('-valid-start-epoch', type=int, default=None,
                       help='meta-train多少个epoch才开始展示meta-valid/meta-test的结果')
    parse.add_argument('-valid-interval', type=int, default=None, help='meta-train多少个epoch才进行一次meta-valid/meta-test')
    parse.add_argument('-valid-draw', type=int, default=None, help='meta-train多少个epoch才进行一绘制一次曲线图')

    # Common
    parse.add_argument('-mode', type=str, default=None, help='训练模式')
    parse.add_argument('-metric', type=str, default=None, help='评估指标名称')
    parse.add_argument('-threshold', type=float, default=None, help='准确率阈值')

    # 训练参数
    # Meta Learning
    parse.add_argument('-backbone', type=str, default=None, help='元学习骨架模型名称')
    parse.add_argument('-if-MIM', type=bool, default=None)
    parse.add_argument('-if-transductive', type=bool, default=None, help='inductive or transductive')
    parse.add_argument('-train-iteration', type=int, default=None, help='meta-train时每个task重复优化多少次')
    parse.add_argument('-valid-iteration', type=int, default=None, help='meta-valid测试多少个任务')
    parse.add_argument('-test-iteration', type=int, default=None, help='meta-test测试多少个任务')
    parse.add_argument('-adapt-iteration', type=int, default=None, help='meta-vald/meta-test时每个task重复优化多少次')
    parse.add_argument('-adapt-lr', type=float, default=None)
    parse.add_argument('-meta-batch-size', type=int, default=None)

    parse.add_argument('-train-way', type=int, default=None)
    parse.add_argument('-train-shot', type=int, default=None)
    parse.add_argument('-train-query', type=int, default=None)
    parse.add_argument('-valid-way', type=int, default=None)
    parse.add_argument('-valid-shot', type=int, default=None)
    parse.add_argument('-valid-query', type=int, default=None)
    parse.add_argument('-test-way', type=int, default=None)
    parse.add_argument('-test-shot', type=int, default=None)
    parse.add_argument('-test-query', type=int, default=None)

    # Common
    parse.add_argument('-model', type=str, default=None, help='元学习模型名称')
    parse.add_argument('-optimizer', type=str, default=None, help='优化器名称')
    parse.add_argument('-loss-func', type=str, default=None, help='损失函数名称, CE/FL')
    parse.add_argument('-epoch', type=int, default=None)
    parse.add_argument('-lr', type=float, default=None)
    parse.add_argument('-reg', type=float, default=None)

    # 损失系数
    # Meta Learning
    parse.add_argument('-alpha', type=float, default=None)
    parse.add_argument('-lamb', type=float, default=None)
    parse.add_argument('-temp', type=float, default=None)

    # Transformer Encoder 模型参数配置
    parse.add_argument('-num-layer', type=int, default=None, help='number of encoder blocks')
    parse.add_argument('-num-head', type=int, default=None, help='number of head in multi-head attention')
    # parse.add_argument('-dim-embedding', type=int, default=None, help='residue embedding dimension')
    parse.add_argument('-dim-feedforward', type=int, default=None, help='hidden layer dimension in feedforward layer')
    parse.add_argument('-dim-k', type=int, default=None, help='embedding dimension of vector k or q')
    parse.add_argument('-dim-v', type=int, default=None, help='embedding dimension of vector v')

    # TextCNN 模型参数
    parse.add_argument('-dim-embedding', type=int, default=None, help='词（残基）向量的嵌入维度')
    parse.add_argument('-dropout', type=float, default=None, help='dropout率')
    parse.add_argument('-static', type=bool, default=None, help='嵌入是否冻结')
    parse.add_argument('-num-filter', type=int, default=None, help='卷积核的数量')
    parse.add_argument('-filter-sizes', type=str, default=None, help='卷积核的尺寸')
    parse.add_argument('-dim-cnn-out', type=int, default=None, help='CNN模型的输出维度')
    parse.add_argument('-output-extend', type=str, default=None, help='CNN后是否再接一层')

    config = parse.parse_args()
    return config
