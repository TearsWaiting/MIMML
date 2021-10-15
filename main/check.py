import pickle
from config import config_SL, config_meta, config_meta_miniImageNet
from Framework import Learner


def check_SL_train():
    config = config_SL.get_config()
    config.learn_name = 'check_SL_train'
    config.mode = 'train-test'
    config.num_class = 2
    config.epoch = 2

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


def check_SL_test():
    config = config_SL.get_config()
    config.learn_name = 'check_SL_test'
    config.mode = 'train-test'
    config.num_class = 2

    learner = Learner.Learner(config)
    learner.setIO('test')
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.test_model()


def check_meta_train():
    config = config_meta.get_config()
    config.learn_name = 'check_meta_train'
    config.mode = 'meta learning'
    config.max_epoch = 2
    config.test_iteration = 2

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


def check_meta_test():
    config = config_meta.get_config()
    config.learn_name = 'check_meta_test'
    config.mode = 'meta learning'
    config.test_iteration = 2

    learner = Learner.Learner(config)
    learner.setIO('test')
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.test_model()


def chcek_SL_pretrain():
    config = config_SL.get_config()
    config.learn_name = 'chcek_SL_pretrain'
    config.path_dataset = '../data/task_data/BPD-36'
    config.epoch = 2
    config.num_class = 28
    config.output_extend = 'pretrain'
    config.alpha = None
    config.batch_size = 256

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.adjust_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def check_meta_train_with_pretrained():
    config = config_meta.get_config()
    config.learn_name = 'check_meta_train_with_pretrained'
    config.path_params = '../result/chcek_SL_pretrain/model/CNN, epoch[1], ACC[0.493].pt'
    # pretrain_config = pickle.load(open('../result/chcek_SL_pretrain/config.pkl', 'rb'))
    config.mode = 'meta learning'
    config.max_epoch = 2
    config.test_iteration = 20

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.adjust_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


def check_meta_finetune():
    # config = pickle.load(open('../result/check_meta_train_with_pretrained/config.pkl', 'rb'))
    config = config_SL.get_config()
    config.learn_name = 'check_meta_finetune'
    config.path_params = '../result/check_meta_train_with_pretrained/model/MIMML, ACC[0.557].pt'
    config.epoch = 2
    config.output_extend = 'pretrain'

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.adjust_model()
    learner.load_params()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.train_model()
    learner.test_model()


if __name__ == '__main__':
    # check_SL_train()
    # check_SL_test()
    # check_meta_train()
    # check_meta_test()

    chcek_SL_pretrain()
    check_meta_train_with_pretrained()
    check_meta_finetune()
