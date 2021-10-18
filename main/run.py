import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pickle
from config import config_SL, config_meta, config_default
from Framework import Learner


def select_fintune_dataset(class_name):
    base_dir = '../data/task_data/Finetune Dataset'
    path_train_data, path_test_data = None, None
    if class_name == 'AAP':
        petide_class_name = '/Anti-angiogenic Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'benchmarkdataset-pospos-train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'benchmarkdataset-pospos-test.tsv'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'NT15dataset-pospos-train.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'NT15dataset-pospos-test.tsv'
    elif class_name == 'ABP':
        petide_class_name = '/Anti-bacterial Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'Main Dataset.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'Independent Dataset.tsv'
    elif class_name == 'ACP':
        petide_class_name = '/Anti-cancer Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'train_main.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'test_main.tsv'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'train_alternate.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'test_alternate.tsv'
    elif class_name == 'AFP':
        petide_class_name = '/Anti-fungal Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'train_main.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'test_main.tsv'
    elif class_name == 'AHP':
        petide_class_name = '/Anti-hypertensive Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'benchmarking.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'Ind.tsv'
    elif class_name == 'AIP':
        petide_class_name = '/Anti-inflammatory Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'AIPpred_train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'AIPpred_test.tsv'
    elif class_name == 'ATP':
        petide_class_name = '/Anti-tubercular Peptide'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'AntiTb_benchmark.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'AntiTb_Ind.tsv'
        path_train_data = base_dir + petide_class_name + '/train/' + 'AntiRD_benchmark.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'AntiRD_Ind.tsv'
    elif class_name == 'AVP':
        petide_class_name = '/Anti-viral Peptide'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'T544p+407n.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'V60p+45n.tsv'
        path_train_data = base_dir + petide_class_name + '/train/' + 'T544p+544n.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'V60p+60n.tsv'
    elif class_name == 'BP':
        petide_class_name = '/Bitter Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'test.tsv'
    elif class_name == 'BBBP':
        petide_class_name = '/Blood–Brain Barrier Penetrating Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'D1_train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'D1_test.tsv'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'D2_train.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'D2_test.tsv'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'D3_train.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'D3_test.tsv'
    elif class_name == 'DPP-IV':
        petide_class_name = '/DPP-IV inhibitory peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'test.tsv'
    elif class_name == 'NP':
        petide_class_name = '/Neuropeptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'training.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'test.tsv'
    elif class_name == 'PSBP':
        petide_class_name = '/Polystyrene Surface-Binding Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'PBP_train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'PBPT_test.tsv'
    elif class_name == 'QSP':
        petide_class_name = '/Quorum Sensing Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'test.tsv'
    elif class_name == 'THP':
        petide_class_name = '/Tumor Homing Peptide'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'Maintrain.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'Maintest.tsv'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'Main90train.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'Main90test.tsv'
        path_train_data = base_dir + petide_class_name + '/train/' + 'Smalltrain.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'Smalltest.tsv'
    elif class_name == 'UP':
        petide_class_name = '/Umami Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'test.tsv'
    elif class_name == 'IL6':
        path_train_data = '../data/task_data/IL-6/Train.tsv'
        path_test_data = '../data/task_data/IL-6/Validate.tsv'
    else:
        print('Error, No Such Dataset')
    return path_train_data, path_test_data


def SL_train(learn_name):
    config = config_SL.get_config()
    if learn_name:
        config.learn_name = learn_name
    else:
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


def SL_multiple_classification(learn_name):
    config = config_SL.get_config()
    if learn_name:
        config.learn_name = learn_name

    else:
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


def few_shot_SL(learn_name):
    # path_train_data, path_test_data = select_fintune_dataset('AAP')
    # path_train_data, path_test_data = select_fintune_dataset('ABP')
    path_train_data, path_test_data = select_fintune_dataset('ACP')
    # path_train_data, path_test_data = select_fintune_dataset('AFP')
    # path_train_data, path_test_data = select_fintune_dataset('AHP')
    # path_train_data, path_test_data = select_fintune_dataset('AIP')
    # path_train_data, path_test_data = select_fintune_dataset('ATP')
    # path_train_data, path_test_data = select_fintune_dataset('AVP')
    # path_train_data, path_test_data = select_fintune_dataset('BP')
    # path_train_data, path_test_data = select_fintune_dataset('BBBP')
    # path_train_data, path_test_data = select_fintune_dataset('DPP-IV')
    # path_train_data, path_test_data = select_fintune_dataset('NP')
    # path_train_data, path_test_data = select_fintune_dataset('PSBP')
    # path_train_data, path_test_data = select_fintune_dataset('QSP')
    # path_train_data, path_test_data = select_fintune_dataset('THP')
    # path_train_data, path_test_data = select_fintune_dataset('UP')
    # path_train_data, path_test_data = select_fintune_dataset('IL6')

    config = config_SL.get_config()
    if learn_name:
        config.learn_name = learn_name
    else:
        config.learn_name = 'inference_AAP'

    config.metric = 'MCC'
    config.output_extend = 'finetune'
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data
    config.device = 1
    config.inference_iteration = 50
    config.dataset = 'inference dataset'
    config.inference_way = 2

    config.epoch = 100
    # config.lr = 0.001
    config.lr = 0.0001

    a = 1
    config.inference_shot = int(a * 689)  # 训练集数量 （正类或负类）
    # config.inference_shot = 5  # 训练集数量 （正类或负类）
    # config.inference_shot = 1  # 训练集数量 （正类或负类）
    config.inference_query = 172  # 测试集数量 （正类或负类）

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.adjust_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.inference('Few-shot SL')


def SL_test(config_path, path_params):
    if config_path:
        config = pickle.load(open(config_path, 'rb'))
    else:
        config = pickle.load(open('../result/SL_train_00/config.pkl', 'rb'))
    if path_params:
        config.path_params = path_params
    else:
        config.path_params = '../result/SL_train_00/model/CNN, epoch[61], ACC[0.918].pt'

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


def SL_fintune(learn_name):
    config = config_SL.get_config()
    if learn_name:
        config.learn_name = learn_name
    else:
        config.learn_name = 'SL_fintune'

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


def pretrain(CMD_config):
    config = config_SL.get_config()
    config.num_meta_train = 24
    config.num_meta_valid = 10
    config.num_meta_test = 10
    config.num_class = config.num_meta_train + 1

    config.metric = 'ACC'
    config.threshold = 0.33
    config.batch_size = 320
    config.output_extend = 'pretrain'
    config.epoch = 50

    print('=' * 50)
    print('The incoming parameters that take effect are:')
    for key, value in CMD_config.__dict__.items():
        if value and key in config.__dict__.keys():
            print('{}: {}'.format(key, value))
            config.__dict__[key] = value
    print('=' * 50)

    for key, value in config.__dict__.items():
        if value is None:
            print('INFO: Value pair is None. [{}]: [{}]'.format(key, value))
    print('=' * 50)

    if config.learn_name is None:
        config.learn_name = 'default_pretrain'
    print('Updated Config', config)

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


def meta_train(CMD_config):
    config = config_meta.get_config()
    print('=' * 50)
    print('The incoming parameters that take effect are:')
    for key, value in CMD_config.__dict__.items():
        if value and key in config.__dict__.keys():
            print('{}: {}'.format(key, value))
            config.__dict__[key] = value
    print('=' * 50)

    for key, value in config.__dict__.items():
        if value is None:
            print('Warning: Value pair is None. [{}]: [{}]'.format(key, value))

    if config.learn_name is None:
        config.learn_name = 'default_meta_train'
    print('Updated Config', config)

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


def meta_train_with_pretrained(CMD_config):
    config = config_meta.get_config()
    config.path_params = '../result/default_pretrain/model/CNN, Epoch[20.000].pt'
    config.threshold = 0.60

    print('=' * 50)
    print('The incoming parameters that take effect are:')
    for key, value in CMD_config.__dict__.items():
        if value and key in config.__dict__.keys():
            print('{}: {}'.format(key, value))
            config.__dict__[key] = value
    print('=' * 50)

    for key, value in config.__dict__.items():
        if value is None:
            print('Warning: Value pair is None. [{}]: [{}]'.format(key, value))

    if config.learn_name is None:
        config.learn_name = 'default_meta_train_with_pretrain'
    print('Updated Config', config)

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


def meta_test(CMD_config):
    if CMD_config.path_config:
        config = pickle.load(open('../result/default_meta_train_with_pretrain/config.pkl', 'rb'))
        print('load existing config')
    else:
        config = config_meta.get_config()

    config.path_params = '../result/default_meta_train_with_pretrain/model/MIMML, Epoch[250.000].pt'

    # config.adapt_iteration = 0
    config.adapt_iteration = 10
    # config.adapt_iteration = 30
    # config.adapt_iteration = 50
    # config.adapt_lr = 0.005
    # config.adapt_lr = 0.001
    config.adapt_lr = 0.0005
    # config.adapt_lr = 0.0001
    # config.adapt_lr = 0.00005

    print('=' * 50)
    print('The incoming parameters that take effect are:')
    for key, value in CMD_config.__dict__.items():
        if value and key in config.__dict__.keys():
            print('{}: {}'.format(key, value))
            config.__dict__[key] = value
    print('=' * 50)

    for key, value in config.__dict__.items():
        if value is None:
            print('Warning: Value pair is None. [{}]: [{}]'.format(key, value))

    if config.learn_name is None:
        config.learn_name = 'default_meta_test'
    print('Updated Config', config)

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


def meta_finetune(CMD_config):
    config = config_SL.get_config()
    config.path_params = '../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt'

    path_train_data, path_test_data = select_fintune_dataset("AAP")
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data
    config.output_extend = 'finetune'
    config.metric = 'MCC'
    config.threshold = 0.48

    config.epoch = 100
    # config.lr = 0.001
    config.lr = 0.0005
    # config.lr = 0.0001
    config.alpha = 0.01
    config.gamma = 2
    config.batch_size = 32

    print('=' * 50)
    print('The incoming parameters that take effect are:')
    for key, value in CMD_config.__dict__.items():
        if value and key in config.__dict__.keys():
            print('{}: {}'.format(key, value))
            config.__dict__[key] = value
    print('=' * 50)

    for key, value in config.__dict__.items():
        if value is None:
            print('Warning: Value pair is None. [{}]: [{}]'.format(key, value))

    if config.learn_name is None:
        config.learn_name = 'default_meta_finetune'
    print('Updated Config', config)

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


def meta_inference(CMD_config):
    if CMD_config.path_config:
        config = pickle.load(open('../result/pretrain_meta_train_BPD_ALL_RT_MIMML/config.pkl', 'rb'))
        print('load existing config')
    else:
        config = config_meta.get_config()

    config.path_params = '../result/pretrain_meta_train_BPD_ALL_RT_MIMML/model/MIMML, Epoch[250.000].pt'

    config.output_extend = 'finetune'
    config.batch_size = 32  # 无用
    config.dim_cnn_out = 128
    config.metric = 'MCC'
    config.threshold = 0.75

    path_train_data, path_test_data = select_fintune_dataset("AAP")
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data

    config.dataset = 'inference dataset'
    config.inference_iteration = 50
    config.inference_way = 2

    a = 0.25
    # config.inference_shot = int(a * 680)  # 训练集数量 （正类或负类）
    config.inference_shot = 5  # 训练集数量 （正类或负类）
    # config.inference_shot = 1  # 训练集数量 （正类或负类）
    config.inference_query = 26  # 测试集数量 （正类或负类）

    config.adapt_iteration = 50
    # config.adapt_iteration = 100

    # config.adapt_lr = 0.005
    # config.adapt_lr = 0.001
    config.adapt_lr = 0.0001
    # config.adapt_lr = 0.00005
    # config.adapt_lr = 0.00001

    print('=' * 50)
    print('The incoming parameters that take effect are:')
    for key, value in CMD_config.__dict__.items():
        if value and key in config.__dict__.keys():
            print('{}: {}'.format(key, value))
            config.__dict__[key] = value
    print('=' * 50)

    for key, value in config.__dict__.items():
        if value is None:
            print('Warning: Value pair is None. [{}]: [{}]'.format(key, value))

    if config.learn_name is None:
        config.learn_name = 'default_meta_inference'
    print('Updated Config', config)

    learner = Learner.Learner(config)
    learner.setIO()
    learner.setVisualization()
    learner.load_data()
    learner.init_model()
    learner.load_params()
    learner.adjust_model()
    learner.init_optimizer()
    learner.def_loss_func()
    learner.inference('Meta Inference')


if __name__ == '__main__':
    config = config_default.get_config()
    print('CMD Config', config)

    if config.task_type_run == "script-test":
        print('Script Test Passed')

    # elif config.task_type_run == "SL-train":
    #     SL_train(config.learn_name)
    # elif config.task_type_run == "SL-multiple-classification":
    #     SL_multiple_classification(config.learn_name)
    # elif config.task_type_run == "few-shot-SL":
    #     few_shot_SL(config.learn_name)
    # elif config.task_type_run == "SL-test":
    #     SL_test(config.config_path, config.path_params)
    # elif config.task_type_run == "SL-finetune":
    #     SL_fintune(config.learn_name)

    elif config.task_type_run == 'pretrain':
        pretrain(config)
    elif config.task_type_run == "meta-train":
        meta_train(config)
    elif config.task_type_run == "meta-train-with-pretrain":
        meta_train_with_pretrained(config)
    elif config.task_type_run == "meta-test":
        meta_test(config)
    elif config.task_type_run == "meta-finetune":
        meta_finetune(config)
    elif config.task_type_run == "meta-inference":
        meta_inference(config)

    else:
        print('Invalid Option Selected')
