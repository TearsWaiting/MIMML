from config import config_SL, config_meta, config_meta_miniImageNet
from Framework import Learner


def SL_fintune(config):
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
    # learner.test_model()


def select_fintune_dataset(class_name):
    base_dir = '../data/task_data/Finetune Dataset'
    path_train_data, path_test_data = None, None
    if class_name == 'AAP':
        petide_class_name = '/Anti-angiogenic Peptide'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'benchmarkdataset-pospos-train.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'benchmarkdataset-pospos-test.tsv'
        path_train_data = base_dir + petide_class_name + '/train/' + 'NT15dataset-pospos-train.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'NT15dataset-pospos-test.tsv'
    elif class_name == 'ABP':
        petide_class_name = '/Anti-bacterial Peptide'
        path_train_data = base_dir + petide_class_name + '/train/' + 'Main Dataset.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'Independent Dataset.tsv'
    elif class_name == 'ACP':
        petide_class_name = '/Anti-cancer Peptide'
        # path_train_data = base_dir + petide_class_name + '/train/' + 'train_main.tsv'
        # path_test_data = base_dir + petide_class_name + '/test/' + 'test_main.tsv'
        path_train_data = base_dir + petide_class_name + '/train/' + 'train_alternate.tsv'
        path_test_data = base_dir + petide_class_name + '/test/' + 'test_alternate.tsv'
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
    else:
        print('Error, No Such Dataset')
    return path_train_data, path_test_data


if __name__ == '__main__':
    config = config_SL.get_config()
    # config.path_params = '../result/pretrain_meta_train_24train_10test_10way_10shot/model/MIMML, Epoch[250.000].pt'
    config.path_params = None
    config.device = 0
    config.output_extend = 'finetune'
    config.metric = 'MCC'
    config.threshold = 0.8

    # config.reg = 0
    # config.alpha = 0.5
    # config.reg = 0.01

    # config.batch_size = 128
    config.batch_size = 32

    # config.epoch = 300
    config.epoch = 100

    # config.lr = 0.00005
    # config.lr = 0.00001
    config.lr = 0.0001
    # config.lr = 0.0005
    # config.lr = 0.001
    # config.lr = 0.005
    # config.lr = 0.01

    # path_train_data, path_test_data = select_fintune_dataset('AAP')
    # path_train_data, path_test_data = select_fintune_dataset('ABP')
    # path_train_data, path_test_data = select_fintune_dataset('ACP')
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
    path_train_data, path_test_data = select_fintune_dataset('THP')
    # path_train_data, path_test_data = select_fintune_dataset('UP')

    config.learn_name = 'fintune_AAP'
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data

    SL_fintune(config)
