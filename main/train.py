import pickle
from config import config_SL, config_meta, config_meta_miniImageNet
from Framework import Learner
from preprocess import meta_dataset


def SL_train():
    config = config_SL.get_config()
    config.learn_name = 'SL_train_00'

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def SL_fintune():
    config = config_SL.get_config()
    # config = pickle.load(open('../result/MIMML_89/config.pkl', 'rb'))
    # config.path_params = '../result/ZZ_BPD36_finetune_MIM_03/model/MIMML, Epoch[50.000].pt'
    # config.path_params = '../result/ZZ_BPD36_finetune_MIM_03/model/MIMML, ACC[0.891].pt'
    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def meta_train():
    config = config_meta.get_config()
    config.learn_name = 'visual_meta_train 0'
    config.path_params = None
    config.path_meta_dataset = '../data/task_data/Meta Dataset/BPD-ALL-RT'

    # config.model_save_name = 'ProtoNet'
    # config.if_MIM = False
    # config.if_transductive = False
    # config.adapt_iteration = 0

    config.model_save_name = 'MIMML'
    config.if_MIM = True
    config.if_transductive = True
    config.adapt_iteration = 10

    config.num_meta_train = 24
    config.num_meta_valid = 10
    config.num_meta_test = 10

    config.train_way = 5
    config.train_shot = 5
    config.train_query = 15
    config.valid_way = 5
    config.valid_shot = 5
    config.valid_query = 15
    config.test_way = 5
    config.test_shot = 5
    config.test_query = 15

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.adjust_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def SL_multiple_classification():
    config = config_SL.get_config()
    config.learn_name = 'pretrain_BPD_ALL_RT'
    config.path_dataset = '../data/task_data/Meta Dataset/BPD-ALL-RT'
    config.alpha = None

    config.num_meta_train = 24
    config.num_meta_valid = 10
    config.num_meta_test = 10

    config.num_class = config.num_meta_train + 1
    config.output_extend = 'pretrain'
    config.threshold = 0.30
    config.batch_size = 320
    config.lr = 0.0002

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.adjust_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def meta_train_with_pretrained():
    config = config_meta.get_config()
    # config.learn_name = 'pretrain_meta_train_BPD_ALL_RT_ProtoNet'
    config.learn_name = 'pretrain_meta_train_24train_10test_10way_1shot'
    config.path_meta_dataset = '../data/task_data/Meta Dataset/BPD-ALL-RT'
    config.path_params = '../result/pretrain_BPD_ALL_RT/model/CNN, epoch[19], ACC[0.371].pt'
    config.threshold = 0.7

    config.num_meta_train = 24
    config.num_meta_valid = 10
    config.num_meta_test = 10

    config.train_way = 10
    config.train_shot = 1
    config.train_query = 15
    config.valid_way = 10
    config.valid_shot = 1
    config.valid_query = 15
    config.test_way = 10
    config.test_shot = 1
    config.test_query = 15

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.adjust_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def meta_finetune():
    config = config_SL.get_config()
    config.learn_name = 'meta_finetune_IL6'
    config.path_params = '../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[100.000].pt'
    # config.path_params = '../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt'
    # config.path_params = '../result/pretrain_meta_train_24train_10test_10way_5shot/model/MIMML, Epoch[250.000].pt'
    config.path_train_data = '../data/task_data/IL-6/Train.tsv'
    config.path_test_data = '../data/task_data/IL-6/Validate.tsv'
    config.output_extend = 'finetune'
    config.metric = 'MCC'
    config.threshold = 0.48

    config.epoch = 100
    # config.lr = 0.001
    # config.lr = 0.0005
    config.lr = 0.0001
    # config.lr = 0.00005
    # config.lr = 0.00001
    # config.reg = 0.01
    config.alpha = 0.01
    config.gamma = 2
    config.batch_size = 32

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.adjust_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


if __name__ == '__main__':
    # SL_train()
    # SL_fintune()
    # train_ProtoNet_miniImageNet()

    # meta_train()
    # SL_multiple_classification()  # 在meta-train set上做监督学习
    meta_train_with_pretrained()  # 在meta-train set上做元学习
    # meta_finetune()  # 在IL-6数据集上进行SL Finetune
