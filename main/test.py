import pickle
import os
from config import config_SL, config_meta, config_meta_miniImageNet
from Framework import Learner


def SL_test():
    config = pickle.load(open('../result/MIMML_peptide_004/config.pkl', 'rb'))
    config.path_params = '../result/MIMML_peptide_004/MIMML, Epoch[200.000].pt'

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    # learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    # learner.train_model()
    learner.test_model()


def test_ProtoNet_miniImageNet():
    config = pickle.load(open('../result/ProtoNet_000/config.pkl', 'rb'))
    config.path_params = '../result/ProtoNet_000/ProtoNet, ACC[0.704].pt'
    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    # learner.train_model()
    learner.test_model()


def meta_test():
    # config = config_meta.get_config()
    # config.learn_name = 'visual_meta_train'
    # config.num_meta_train = 24
    # config.num_meta_valid = 10
    # config.num_meta_test = 10

    # Pretrain + MIMML
    # config = pickle.load(open('../result/pretrain_meta_train_BPD_ALL_RT_MIMML/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[0.000].pt'
    # config.path_params = None

    # Pretrain + ProtoNet
    # config = pickle.load(open('../result/pretrain_meta_train_BPD_ALL_RT_ProtoNet/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_BPD_ALL_RT_ProtoNet/model/ProtoNet, Epoch[0.000].pt'
    # config.path_params = None

    # 24 meta-train, 10 meta-test, 5-way, 10-shot, 15-query
    # config = pickle.load(open('../result/pretrain_meta_train_24train_10test_5way_10shot/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_24train_10test_5way_10shot/model/MIMML, Epoch[130.000].pt'

    # 24 meta-train, 10 meta-test, 5-way, 5-shot, 15-query
    # config = pickle.load(open('../result/pretrain_meta_train_24train_10test_5way_5shot/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_24train_10test_5way_5shot/model/MIMML, Epoch[100.000].pt'

    # 24 meta-train, 10 meta-test, 5-way, 1-shot, 15-query
    # config = pickle.load(open('../result/pretrain_meta_train_24train_10test_5way_1shot/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_24train_10test_5way_1shot/model/MIMML, Epoch[200.000].pt'

    # 24 meta-train, 10 meta-test, 10-way, 10-shot, 15-query
    # config = pickle.load(open('../result/pretrain_meta_train_24train_10test_10way_10shot/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_24train_10test_10way_10shot/model/MIMML, Epoch[40.000].pt'

    # 24 meta-train, 10 meta-test, 10-way, 5-shot, 15-query
    # config = pickle.load(open('../result/pretrain_meta_train_24train_10test_10way_5shot/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_24train_10test_10way_5shot/model/MIMML, Epoch[200.000].pt'

    # 16 meta-train, 10 meta-test, 5-way, 5-shot, 15-query
    # config = pickle.load(open('../result/pretrain_meta_train_16train_10test_5way_5shot/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_16train_10test_5way_5shot/model/MIMML, Epoch[200.000].pt'

    # 16 meta-train, 10 meta-test, 5-way, 1-shot, 15-query
    # config = pickle.load(open('../result/pretrain_meta_train_16train_10test_5way_1shot/config.pkl', 'rb'))
    # config.path_params = '../result/pretrain_meta_train_16train_10test_5way_1shot/model/MIMML, Epoch[200.000].pt'

    # ProtoNet 24 meta-train, 10 meta-test, 5-way, 1-shot, 15-query
    # config = pickle.load(open('../result/ProtoNet_24train_10test_5way_1shot/config.pkl', 'rb'))
    # config.path_params = '../result/ProtoNet_24train_10test_5way_1shot/model/ProtoNet, Epoch[150.000].pt'

    # visual
    config = pickle.load(open('../result/visual_meta_train/config.pkl', 'rb'))
    config.path_params = '../result/visual_meta_train/model/MIMML, Epoch[200.000].pt'
    # config.path_params = '../result/visual_meta_train/model/ProtoNet, Epoch[200.000].pt'

    config.device = 0
    # config.test_iteration = 50
    # config.if_MIM = False
    # config.if_transductive = False
    # config.if_MIM = True
    # config.if_transductive = True

    # config.adapt_iteration = 0
    # config.adapt_iteration = 10
    # config.adapt_iteration = 50
    # config.adapt_lr = 0.005
    # config.adapt_lr = 0.001
    # config.adapt_lr = 0.0005
    # config.adapt_lr = 0.0001
    # config.adapt_lr = 0.00005

    learner = Learner.Learner(config)
    learner.setIO('test')
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.adjust_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.test_model()


if __name__ == '__main__':
    # SL_test()
    # test_ProtoNet_miniImageNet()
    meta_test()
