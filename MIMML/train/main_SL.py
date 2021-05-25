# ---encoding:utf-8---
# @Time : 2021.03.02
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : main_SL.py

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from preprocess import data_loader, data_loader_kmer
from configuration import config as cf
from util import util_metric
from model import BERT
from train.model_operation import save_model, adjust_model
from train.visualization import dimension_reduction, penultimate_feature_visulization

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import pickle
import seaborn as sns


def load_data(config):
    if_multi_scaled = config.if_multi_scaled

    if if_multi_scaled:
        residue2idx = pickle.load(open('../data/kmer_residue2idx.pkl', 'rb'))
        config.vocab_size = len(residue2idx)
        config.token2index = residue2idx
        print('old config.vocab_size:', config.vocab_size)

        train_iter_orgin, test_iter = data_loader_kmer.load_data(config)
        config.max_len = 256
    else:
        residue2idx = pickle.load(open('../data/meta_data/residue2idx.pkl', 'rb'))
        config.vocab_size = len(residue2idx)
        config.token2index = residue2idx

        train_iter_orgin, test_iter = data_loader.load_data(config)

    print('-' * 20, 'data construction over', '-' * 20)
    print('config.vocab_size', config.vocab_size)
    print('max_len_train', config.max_len_train)
    print('max_len_test', config.max_len_test)
    print('config.max_len', config.max_len)
    return train_iter_orgin, test_iter


def draw_figure_CV(config, fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(train_loss_record):
        train_loss_record[i] = e.cpu().detach()

    for i, e in enumerate(valid_acc_record):
        valid_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(valid_loss_record):
        valid_loss_record[i] = e.cpu().detach()

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Validation Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_valid_interval, valid_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Validation Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_valid_interval, valid_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


def draw_figure_train_test(config, fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, e in enumerate(train_acc_record):
        train_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(train_loss_record):
        train_loss_record[i] = e.cpu().detach()

    for i, e in enumerate(test_acc_record):
        test_acc_record[i] = e.cpu().detach()

    for i, e in enumerate(test_loss_record):
        test_loss_record[i] = e.cpu().detach()

    plt.subplot(2, 2, 1)
    plt.title("Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_log_interval, train_acc_record)
    plt.subplot(2, 2, 2)
    plt.title("Train Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_log_interval, train_loss_record)
    plt.subplot(2, 2, 3)
    plt.title("Test Acc Curve", fontsize=23)
    plt.xlabel("Epoch", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_test_interval, test_acc_record)
    plt.subplot(2, 2, 4)
    plt.title("Test Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_test_interval, test_loss_record)

    plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


def cal_loss_dist_by_cosine(model):
    embedding = model.embedding
    loss_dist = 0

    vocab_size = embedding[0].tok_embed.weight.shape[0]
    d_model = embedding[0].tok_embed.weight.shape[1]

    Z_norm = vocab_size * (len(embedding) ** 2 - len(embedding)) / 2

    for i in range(len(embedding)):
        for j in range(len(embedding)):
            if i < j:
                cosin_similarity = torch.cosine_similarity(embedding[i].tok_embed.weight, embedding[j].tok_embed.weight)
                loss_dist -= torch.sum(cosin_similarity)
                # print('cosin_similarity.shape', cosin_similarity.shape)
    loss_dist = loss_dist / Z_norm
    return loss_dist


def get_loss(logits, label, criterion):
    loss = criterion(logits.view(-1, config.num_class), label.view(-1))
    loss = (loss.float()).mean()

    # flooding method
    loss = (loss - config.b).abs() + config.b

    # multi-sense loss
    # alpha = -0.1
    # loss_dist = alpha * cal_loss_dist_by_cosine(model)
    # loss += loss_dist

    return loss


def periodic_test(test_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Test' + '#' * 60)
    test_metric, test_loss, test_repres_list, test_label_list, \
    test_roc_data, test_prc_data = model_eval(test_iter, model, criterion, config)

    print('test current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(test_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_test_interval.append(sum_epoch)
    test_acc_record.append(test_metric[0])
    test_loss_record.append(test_loss)

    return test_metric, test_loss, test_repres_list, test_label_list


def periodic_valid(valid_iter, model, criterion, config, sum_epoch):
    print('#' * 60 + 'Periodic Validation' + '#' * 60)

    valid_metric, valid_loss, valid_repres_list, valid_label_list, \
    valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)

    print('validation current performance')
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(valid_metric.numpy())
    print('#' * 60 + 'Over' + '#' * 60)

    step_valid_interval.append(sum_epoch)
    valid_acc_record.append(valid_metric[0])
    valid_loss_record.append(valid_loss)

    return valid_metric, valid_loss, valid_repres_list, valid_label_list


def train_ACP(train_iter, valid_iter, test_iter, model, optimizer, criterion, config, iter_k):
    steps = 0
    best_acc = 0
    best_performance = 0

    for epoch in range(1, config.epoch + 1):
        repres_list = []
        label_list = []

        for batch in train_iter:
            if config.if_multi_scaled:
                input, origin_input, label = batch
                logits, output = model(input, origin_input)
            else:
                input, label = batch
                logits, output = model(input)
                # print('output.size()', output.size())

                repres_list.extend(output.cpu().detach().numpy())
                label_list.extend(label.cpu().detach().numpy())

            loss = get_loss(logits, label, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

            '''Periodic Train Log'''
            if steps % config.interval_log == 0:
                corrects = (torch.max(logits, 1)[1] == label).sum()
                the_batch_size = label.shape[0]
                train_acc = 100.0 * corrects / the_batch_size
                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, steps,
                                                                                        loss,
                                                                                        train_acc,
                                                                                        corrects,
                                                                                        the_batch_size))
                print()

                step_log_interval.append(steps)
                train_acc_record.append(train_acc)
                train_loss_record.append(loss)

        sum_epoch = iter_k * config.epoch + epoch

        '''Periodic Validation'''
        if valid_iter and sum_epoch % config.interval_valid == 0:
            valid_metric, valid_loss, valid_repres_list, valid_label_list = periodic_valid(valid_iter,
                                                                                           model,
                                                                                           criterion,
                                                                                           config,
                                                                                           sum_epoch)
            valid_acc = valid_metric[0]
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_performance = valid_metric

        '''Periodic Test'''
        if test_iter and sum_epoch % config.interval_test == 0:
            time_test_start = time.time()

            test_metric, test_loss, test_repres_list, test_label_list = periodic_test(test_iter,
                                                                                      model,
                                                                                      criterion,
                                                                                      config,
                                                                                      sum_epoch)
            '''Periodic Save'''
            # save the model if specific conditions are met
            test_acc = test_metric[0]
            if test_acc > best_acc:
                best_acc = test_acc
                best_performance = test_metric
                if config.save_best and best_acc > config.threshold:
                    save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)

            test_label_list = [x + 2 for x in test_label_list]
            repres_list.extend(test_repres_list)
            label_list.extend(test_label_list)

            '''feature dimension reduction'''
            # if sum_epoch % 1 == 0 or epoch == 1:
            #     dimension_reduction(repres_list, label_list, epoch)

            '''reduction feature visualization'''
            # if sum_epoch % 5 == 0 or epoch == 1 or (epoch % 2 == 0 and epoch <= 10):
            #     penultimate_feature_visulization(repres_list, label_list, epoch)
            #
            # time_test_end = time.time()
            # print('inference time:', time_test_end - time_test_start, 'seconds')

    return best_performance


def model_eval(data_iter, model, criterion, config):
    device = torch.device("cuda" if config.cuda else "cpu")
    label_pred = torch.empty([0], device=device)
    label_real = torch.empty([0], device=device)
    pred_prob = torch.empty([0], device=device)

    print('model_eval data_iter', len(data_iter))

    iter_size, corrects, avg_loss = 0, 0, 0
    repres_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for batch in data_iter:
            if config.if_multi_scaled:
                input, origin_inpt, label = batch
                logits, output = model(input, origin_inpt)
            else:
                input, label = batch
                logits, output = model(input)

            repres_list.extend(output.cpu().detach().numpy())
            label_list.extend(label.cpu().detach().numpy())

            loss = criterion(logits.view(-1, config.num_class), label.view(-1))
            loss = (loss.float()).mean()
            avg_loss += loss

            pred_prob_all = F.softmax(logits, dim=1)
            # Prediction probability [batch_size, class_num]
            pred_prob_positive = pred_prob_all[:, 1]
            # Probability of predicting positive classes [batch_size]
            pred_prob_sort = torch.max(pred_prob_all, 1)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            # The location (class) of the predicted maximum probability in each sample [batch_size]
            corrects += (pred_class == label).sum()

            iter_size += label.shape[0]

            label_pred = torch.cat([label_pred, pred_class.float()])
            label_real = torch.cat([label_real, label.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

    metric, roc_data, prc_data = util_metric.caculate_metric(label_pred, label_real, pred_prob)
    avg_loss /= iter_size
    # accuracy = 100.0 * corrects / iter_size
    accuracy = metric[0]
    print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_loss,
                                                                  accuracy,
                                                                  corrects,
                                                                  iter_size))

    return metric, avg_loss, repres_list, label_list, roc_data, prc_data


def k_fold_CV(train_iter_orgin, test_iter, config):
    valid_performance_list = []

    for iter_k in range(config.k_fold):
        print('=' * 50, 'iter_k={}'.format(iter_k + 1), '=' * 50)

        # Cross validation on training set
        train_iter = [x for i, x in enumerate(train_iter_orgin) if i % config.k_fold != iter_k]
        valid_iter = [x for i, x in enumerate(train_iter_orgin) if i % config.k_fold == iter_k]
        print('----------Data Selection----------')
        print('train_iter index', [i for i, x in enumerate(train_iter_orgin) if i % config.k_fold != iter_k])
        print('valid_iter index', [i for i, x in enumerate(train_iter_orgin) if i % config.k_fold == iter_k])

        print('len(train_iter_orgin)', len(train_iter_orgin))
        print('len(train_iter)', len(train_iter))
        print('len(valid_iter)', len(valid_iter))
        if test_iter:
            print('len(test_iter)', len(test_iter))
        print('----------Data Selection Over----------')

        if config.model_name == 'BERT':
            model = BERT.BERT(config)

        if config.cuda: model.cuda()
        adjust_model(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.reg)
        criterion = nn.CrossEntropyLoss()
        model.train()

        print('=' * 50 + 'Start Training' + '=' * 50)
        valid_performance = train_ACP(train_iter, valid_iter, test_iter, model, optimizer, criterion, config, iter_k)
        print('=' * 50 + 'Train Finished' + '=' * 50)

        print('=' * 40 + 'Cross Validation iter_k={}'.format(iter_k + 1), '=' * 40)
        valid_metric, valid_loss, valid_repres_list, valid_label_list, \
        valid_roc_data, valid_prc_data = model_eval(valid_iter, model, criterion, config)
        print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print(valid_metric.numpy())
        print('=' * 40 + 'Cross Validation Over' + '=' * 40)

        valid_performance_list.append(valid_performance)

        '''draw figure'''
        draw_figure_CV(config, config.learn_name + '_k[{}]'.format(iter_k + 1))

        '''reset plot data'''
        global step_log_interval, train_acc_record, train_loss_record, \
            step_valid_interval, valid_acc_record, valid_loss_record
        step_log_interval = []
        train_acc_record = []
        train_loss_record = []
        step_valid_interval = []
        valid_acc_record = []
        valid_loss_record = []

    return model, valid_performance_list


def train_test(train_iter, test_iter, config):
    print('=' * 50, 'train-test', '=' * 50)
    print('len(train_iter)', len(train_iter))
    print('len(test_iter)', len(test_iter))

    if config.model_name == 'BERT':
        print('model_name', config.model_name)
        model = BERT.BERT(config)

    if config.cuda: model.cuda()
    adjust_model(model)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=config.reg)
    criterion = nn.CrossEntropyLoss()
    # model.train()

    print('=' * 50 + 'Start Training' + '=' * 50)
    best_performance = train_ACP(train_iter, None, test_iter, model, optimizer, criterion, config, 0)
    print('=' * 50 + 'Train Finished' + '=' * 50)

    print('*' * 60 + 'The Last Test' + '*' * 60)
    last_test_metric, last_test_loss, last_test_repres_list, last_test_label_list, \
    last_test_roc_data, last_test_prc_data = model_eval(test_iter, model, criterion, config)
    print('[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
    print(last_test_metric.numpy())
    print('*' * 60 + 'The Last Test Over' + '*' * 60)

    return model, best_performance, last_test_metric


def select_dataset():
    # ACP dataset
    # path_train_data = '../data/task_data/ACP_dataset/tsv/ACP2_main_train.tsv'
    # path_test_data = '../data/task_data/ACP_dataset/tsv/ACP2_main_test.tsv'

    # path_train_data = '../data/task_data/ACP/ACP-Mixed-100-train.tsv'
    # path_test_data = '../data/task_data/ACP/ACP-Mixed-100-test.tsv'
    # path_train_data = '../data/task_data/ACP/ACP-Mixed-80-train.tsv'
    # path_test_data = '../data/task_data/ACP/ACP-Mixed-80-test.tsv'
    # path_train_data = '../data/task_data/ACP/ACP2_main_train.tsv'
    # path_test_data = '../data/task_data/ACP/ACP2_main_test.tsv'
    # path_train_data = '../data/task_data/ACP/ACP2_alternate_train.tsv'
    # path_test_data = '../data/task_data/ACP/ACP2_alternate_test.tsv'

    # Functional Peptides finetune dataset
    # path_train_data = '../data/task_data/finetune dataset/Anti-angiogenic Peptides/benchmarkdataset.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-angiogenic Peptides/NT15dataset.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-cancer Peptides/ACP2_main_train.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-cancer Peptides/ACP2_main_test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-cancer Peptides/ACP2_alternate_train.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-cancer Peptides/ACP2_alternate_test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-fungal Peptides/AntiFP_main.tsv'
    # path_test_data = None
    # path_train_data = '../data/task_data/finetune dataset/Anti-fungal Peptides/Training_antifngl_DS1.tsv'
    # path_test_data = None
    # path_train_data = '../data/task_data/finetune dataset/Anti-fungal Peptides/AntiFP_main.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-fungal Peptides/independent_main1.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-inflammatory Peptides/training.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-inflammatory Peptides/test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-tubercular Peptides/AntiRD_benchmark.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-tubercular Peptides/AntiRD_Ind.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-tubercular Peptides/AntiTb_benchmark.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-tubercular Peptides/AntiTb_Ind.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-viral Peptides/T544p+407n.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-viral Peptides/V60p+45n.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-viral Peptides/T544p+544n.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-viral Peptides/V60p+60n.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Bitter Peptides/training.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Bitter Peptides/test.tsv'
    path_train_data = '../data/task_data/finetune dataset/Cell-penetrating Peptides/training.tsv'
    path_test_data = '../data/task_data/finetune dataset/Cell-penetrating Peptides/test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/DPP IV Inhibitor/training.tsv'
    # path_test_data = '../data/task_data/finetune dataset/DPP IV Inhibitor/test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Neuropeptides/training_ds1.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Neuropeptides/test_ds1.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Neuropeptides/training_ds2.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Neuropeptides/test_ds2.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Neuropeptides/training.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Neuropeptides/test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Quorum Sensing Peptides/training.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Quorum Sensing Peptides/test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Surface-binding Peptides/dataset.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Surface-binding Peptides/dataset.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Tumor Homing Peptides/main.tsv'
    # path_test_data = None
    # path_train_data = '../data/task_data/finetune dataset/Tumor Homing Peptides/main90.tsv'
    # path_test_data = None
    # path_train_data = '../data/task_data/finetune dataset/Tumor Homing Peptides/small.tsv'
    # path_test_data = None
    # path_train_data = '../data/task_data/finetune dataset/Umami Peptides/training.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Umami Peptides/test.tsv'

    return path_train_data, path_test_data


def load_config():
    '''The following variables need to be actively determined for each training session:
       1.train-name: Name of the training
       2.path-config-data: The path of the model configuration. 'None' indicates that the default configuration is loaded
       3.path-train-data: The path of training set
       4.path-test-data: Path to test set

       Each training corresponds to a result folder named after train-name, which contains:
       1.report: Training report
       2.figure: Training figure
       3.config: model configuration
       4.model_save: model parameters
       5.others: other data
       '''

    '''Set the required variables in the configuration'''
    train_name = 'ACPred-LAF'
    path_config_data = None
    path_train_data, path_test_data = select_dataset()

    '''Get configuration'''
    if path_config_data is None:
        config = cf.get_train_config()
    else:
        config = pickle.load(open(path_config_data, 'rb'))

    '''Modify default configuration'''
    # config.epoch = 50
    # config.batch_size = 64

    '''Set other variables'''
    # flooding method
    b = 0.06
    model_name = 'BERT'

    '''initialize result folder'''
    result_folder = '../result/' + config.learn_name
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    '''Save all variables in configuration'''
    config.train_name = train_name
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data

    config.b = b
    config.if_multi_scaled = False
    config.model_name = model_name
    config.result_folder = result_folder

    return config


if __name__ == '__main__':
    np.set_printoptions(linewidth=400, precision=4)
    time_start = time.time()

    '''load configuration'''
    config = load_config()

    '''set device'''
    torch.cuda.set_device(config.device)

    '''load data'''
    train_iter, test_iter = load_data(config)
    print('=' * 20, 'load data over', '=' * 20)

    '''draw preparation'''
    step_log_interval = []
    train_acc_record = []
    train_loss_record = []
    step_valid_interval = []
    valid_acc_record = []
    valid_loss_record = []
    step_test_interval = []
    test_acc_record = []
    test_loss_record = []

    '''train procedure'''
    valid_performance = 0
    best_performance = 0
    last_test_metric = 0

    if config.k_fold == -1:
        # train and test
        model, best_performance, last_test_metric = train_test(train_iter, test_iter, config)
        pass
    else:
        # k cross validation
        model, valid_performance_list = k_fold_CV(train_iter, None, config)

    '''draw figure'''
    draw_figure_train_test(config, config.learn_name)

    '''report result'''
    print('*=' * 50 + 'Result Report' + '*=' * 50)
    if config.k_fold != -1:
        print('valid_performance_list', valid_performance_list)
        tensor_list = [x.view(1, -1) for x in valid_performance_list]
        cat_tensor = torch.cat(tensor_list, dim=0)
        metric_mean = torch.mean(cat_tensor, dim=0)

        print('valid mean performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(metric_mean.numpy()))

        print('valid_performance list')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        for tensor_metric in valid_performance_list:
            print('\t{}'.format(tensor_metric.numpy()))
    else:
        print('last test performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(last_test_metric))
        print()
        print('best_performance')
        print('\t[ACC,\tPrecision,\tSensitivity,\tSpecificity,\tF1,\tAUC,\tMCC]')
        print('\t{}'.format(best_performance))

    print('*=' * 50 + 'Report Over' + '*=' * 50)

    '''save train result'''
    # save the model if specific conditions are met
    if config.k_fold == -1:
        best_acc = best_performance[0]
        last_test_acc = last_test_metric[0]
        if last_test_acc > best_acc:
            best_acc = last_test_acc
            best_performance = last_test_metric
            if config.save_best and best_acc >= config.threshold:
                save_model(model.state_dict(), best_acc, config.result_folder, config.learn_name)

    # save the model configuration
    with open(config.result_folder + '/Trainp[10], Test[10], Epoch[30], ACC[0.5420].pkl', 'wb') as file:
        pickle.dump(config, file)
    print('-' * 50, 'Config Save Over', '-' * 50)

    time_end = time.time()
    print('total time cost', time_end - time_start, 'seconds')
