# ---encoding:utf-8---
# @Time : 2021.03.30
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : MIMML_sequence.py


import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import learn2learn as l2l
import numpy as np
import time
import pickle
import os
import random

from copy import deepcopy
from learn2learn.data.transforms import RemapLabels
from train.model_operation import save_model, load_model
from preprocess import FPD_process
from model import Transformer_Encoder
from tqdm import tqdm
from matplotlib.colors import ListedColormap


def draw_decision_boundary(X, y, proto_embeddings, config, resolution=0.002):
    X_support, X_query = X
    y_support, y_query = y

    X_support = X_support.cpu().detach().numpy()
    X_query = X_query.cpu().detach().numpy()
    y_support = y_support.cpu().detach().numpy()
    y_query = y_query.cpu().detach().numpy()

    plt.figure(11, figsize=(8, 6))
    # plt.figure(11, figsize=(8, 6), dpi=300)
    # markers = ('s', 'x', 'o', '^', 'v')
    markers = ('o', '^', '*')
    colors = ('blue', 'red', 'lightgreen', 'orange', 'purple')
    cmap = ListedColormap(colors[:len(np.unique(y_support))])

    # plot the decision surface
    x1_min_support, x1_max_support = X_support[:, 0].min() - 0.1 - 0.3, X_support[:, 0].max() + 0.1
    x2_min_support, x2_max_support = X_support[:, 1].min() - 0.1, X_support[:, 1].max() + 0.1
    x1_min_query, x1_max_query = X_query[:, 0].min() - 0.1 - 0.3, X_query[:, 0].max() + 0.1
    x2_min_query, x2_max_query = X_query[:, 1].min() - 0.1, X_query[:, 1].max() + 0.1
    x1_min = min(x1_min_support, x1_min_query)
    x1_max = max(x1_max_support, x1_max_query)
    x2_min = min(x2_min_support, x2_min_query)
    x2_max = max(x2_max_support, x2_max_query)

    points_x1, points_x2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                       np.arange(x2_min, x2_max, resolution))

    points = np.array([points_x1.ravel(), points_x2.ravel()]).T
    device = torch.device('cuda') if config.cuda else torch.device('cpu')
    points = torch.from_numpy(points)

    points = points.to(device)
    points_logits = pairwise_distances_logits(points, proto_embeddings, config.temp)
    pred = points_logits.argmax(dim=1).view(points_logits.size(0))

    Z = pred.reshape(points_x1.shape)
    Z = Z.cpu().detach().numpy()
    plt.contourf(points_x1, points_x2, Z, 5, alpha=0.3, cmap=cmap)

    plt.xlim(points_x1.min(), points_x1.max())
    plt.ylim(points_x2.min(), points_x2.max())

    # plot class samples
    for idx, label in enumerate(np.unique(y_support)):
        plt.scatter(x=X_support[y_support == label, 0],
                    y=X_support[y_support == label, 1],
                    s=100,
                    alpha=0.6,
                    c=colors[idx],
                    marker=markers[0],
                    label=label,
                    edgecolors='black')

    for idx, label in enumerate(np.unique(y_query)):
        plt.scatter(x=X_query[y_query == label, 0],
                    y=X_query[y_query == label, 1],
                    s=80,
                    alpha=0.6,
                    c=colors[idx],
                    marker=markers[1],
                    label=label,
                    edgecolors='black')

    prototype = proto_embeddings.cpu().detach().numpy()
    for idx, p in enumerate(prototype):
        plt.scatter(x=p[0],
                    y=p[1],
                    s=150,
                    alpha=0.9,
                    c=colors[idx],
                    marker=markers[2],
                    label=idx,
                    edgecolors='black')

    plt.xticks(fontproperties='Times New Roman', size=13)
    plt.yticks(fontproperties='Times New Roman', size=13)
    plt.xlabel('Dimension 1', fontsize=15)
    plt.ylabel('Dimension 2', fontsize=15)
    plt.legend(loc='upper left')
    font = {"color": "darkred", "size": 18, "family": "serif"}
    plt.title("{} Visualization".format(config.title), fontdict=font)
    plt.show()


def draw_figure(fig_name):
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    # for i in range(len(step_list)):
    #     train_loss_list[i] = train_loss_list[i].cpu().detach().numpy()
    #     valid_loss_list[i] = valid_loss_list[i].cpu().detach().numpy()
    #     train_accuracy_list[i] = train_accuracy_list[i].cpu().detach().numpy()
    #     valid_accuracy_list[i] = valid_accuracy_list[i].cpu().detach().numpy()

    plt.subplot(2, 2, 1)
    plt.title("Average Train Evaluation Loss", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_list, train_loss_list)
    plt.subplot(2, 2, 2)
    plt.title("Average Valid Evaluation Loss", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_list, valid_loss_list)
    plt.subplot(2, 2, 3)
    plt.title("Average Train Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_list, train_accuracy_list)
    plt.subplot(2, 2, 4)
    plt.title("Average Valid Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_list, valid_accuracy_list)

    save_dir = config.path_save + config.learn_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name + '.png')
    plt.show()


def get_MI(probs):
    cond_ent = get_cond_entropy(probs)
    ent = get_entropy(probs)
    return ent - cond_ent


def get_entropy(probs):
    ent = - (probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
    return ent


def get_cond_entropy(probs):
    cond_ent = - (probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
    return cond_ent


def pairwise_distances_logits(a, b, temperature):
    n = a.shape[0]
    m = b.shape[0]
    logits = -0.5 * temperature * ((a.unsqueeze(1).expand(n, m, -1) -
                                    b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(model, task, way, shot, query_num, device):
    data, labels = task
    data = [x.view(-1, 1) for x in data]
    data = torch.cat(data, dim=1)
    data = data.to(device)

    reset_labels = np.repeat(np.arange(way), shot + query_num, 0)
    reset_labels = torch.from_numpy(reset_labels)
    labels = reset_labels.to(device)

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)[1]

    # 对embedding进行归一化
    embeddings = F.normalize(embeddings, dim=1)

    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(way) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)
    support_embeddings = embeddings[support_indices]
    proto_embeddings = support_embeddings.reshape(way, shot, -1).mean(dim=1)
    query_embeddings = embeddings[query_indices]
    query_labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query_embeddings, proto_embeddings, 1)
    loss = F.cross_entropy(logits, query_labels)
    acc = accuracy(logits, query_labels)

    # 绘图
    # support_labels = labels[support_indices].long()
    # config.title = 'ProtoNet'
    # if epoch % 5 == 0 and task_id == 99:
    #     draw_embeddings = [support_embeddings, query_embeddings]
    #     draw_labels = [support_labels, query_labels]
    #     draw_decision_boundary(draw_embeddings, draw_labels, proto_embeddings, config)

    return loss, acc


def fast_adapt_MIM(model, task, way, shot, query_num, device, config, if_train):
    data, labels = task
    data = [x.view(-1, 1) for x in data]
    data = torch.cat(data, dim=1)
    data = data.to(device)

    # print('data', data.size(), data)
    # print('labels', labels.size(), labels)

    reset_labels = np.repeat(np.arange(way), shot + query_num, 0)
    reset_labels = torch.from_numpy(reset_labels)
    labels = reset_labels.to(device)
    # print('reset labels', labels.size(), labels)

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # Compute support and query embeddings
    embeddings = model(data)[1]
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(way) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)

    # 对embedding进行归一化
    embeddings = F.normalize(embeddings, dim=1)

    # 计算 embeddings
    support_embeddings = embeddings[support_indices]
    # support_embeddings: [shot, dim_feature]
    proto_embeddings = support_embeddings.reshape(way, shot, -1).mean(dim=1)
    # proto_embeddings: [way, dim_feature]
    query_embeddings = embeddings[query_indices]
    # support: [query_num, dim_feature]

    # 对embedding进行归一化
    # support_embeddings = F.normalize(support_embeddings.view(support_embeddings.size(0), -1), dim=1)
    # proto_embeddings = F.normalize(proto_embeddings.view(proto_embeddings.size(0), -1), dim=1)
    # query_embeddings = F.normalize(query_embeddings.view(query_embeddings.size(0), -1), dim=1)

    # 计算 labels
    support_labels = labels[support_indices].long()
    # query_labels: [shot]
    query_labels = labels[query_indices].long()
    # query_labels: [query_num]

    # 计算 logits
    support_logits = pairwise_distances_logits(support_embeddings, proto_embeddings, config.temp)
    # support_logits: [shot, way]
    query_logits = pairwise_distances_logits(query_embeddings, proto_embeddings, config.temp)
    # query_logits: [query_num, way]

    # 计算 CE loss
    loss_support_CE = (F.cross_entropy(support_logits, support_labels).float()).mean()

    if if_train:
        loss_query_CE = (F.cross_entropy(query_logits, query_labels).float()).mean()

    # 计算 ACC
    support_acc = accuracy(support_logits, support_labels)
    query_acc = accuracy(query_logits, query_labels)

    # softmax
    support_probs = support_logits.softmax(1)
    query_probs = query_logits.softmax(1)

    # 计算 Entropy, Conditional Entropy and Mutual Information
    support_ent = get_entropy(probs=support_probs)
    support_cond_ent = get_cond_entropy(probs=support_probs)
    # support_mi = support_ent - support_cond_ent

    query_ent = get_entropy(probs=query_probs)
    query_cond_ent = get_cond_entropy(probs=query_probs)
    query_mi = query_ent - query_cond_ent

    # 计算总损失
    if if_train:
        '''L_S_CE + L_Q_CE + L_S_MI + L_Q_MI'''
        loss_sum = config.lamb * (loss_support_CE + loss_query_CE) - \
                   (support_ent - config.alpha * support_cond_ent) - \
                   (query_ent - config.alpha * query_cond_ent)

        '''L_S_CE + L_Q_CE + L_Q_MI'''
        # loss_sum = config.lamb * (loss_support_CE + loss_query_CE) + \
        #            (query_ent - config.alpha * query_cond_ent)

        '''L_S_CE + L_Q_CE + L_S_MI'''
        # loss_sum = config.lamb * (loss_support_CE + loss_query_CE) + \
        #            (support_ent - config.alpha * support_cond_ent)

        '''L_S_CE + L_Q_MI'''
        # loss_sum = config.lamb * loss_support_CE - (query_ent - config.alpha * query_cond_ent)

        '''L_S_CE + L_Q_CE '''
        # loss_sum = config.lamb * (loss_support_CE + loss_query_CE)

        '''L_S_MI + L_Q_MI'''
        # loss_sum = - (support_ent - config.alpha * support_cond_ent) \
        #            - (query_ent - config.alpha * query_cond_ent)

        ''''L_S_CE + L_S_MI + L_Q_MI'''
        # loss_sum = config.lamb * (loss_support_CE) \
        #            - (support_ent - config.alpha * support_cond_ent) \
        #            - (query_ent - config.alpha * query_cond_ent)

        ''''L_Q_CE + L_S_MI + L_Q_MI'''
        # loss_sum = config.lamb * (loss_query_CE) \
        #            - (support_ent - config.alpha * support_cond_ent) \
        #            - (query_ent - config.alpha * query_cond_ent)

        '''L_S_MI'''
        # loss_sum = - (support_ent - config.alpha * support_cond_ent)

        '''L_Q_MI'''
        # loss_sum = - (query_ent - config.alpha * query_cond_ent)

        '''L_S_CE'''
        # loss_sum = config.lamb * loss_support_CE

        ''''L_Q_CE'''
        # loss_sum = config.lamb * loss_query_CE

    else:
        '''L_S_CE + L_S_MI + L_Q_MI'''
        loss_sum = config.lamb * (loss_support_CE) - \
                   (support_ent - config.alpha * support_cond_ent) - \
                   (query_ent - config.alpha * query_cond_ent)

    # 绘图
    # config.title = 'MIMML'
    # if if_train and epoch % 5 == 0 and task_id == 99:
    #     draw_embeddings = [support_embeddings, query_embeddings]
    #     draw_labels = [support_labels, query_labels]
    #     draw_decision_boundary(draw_embeddings, draw_labels, proto_embeddings, config)
    # if not if_train:
    #     if task_id == 1 and j == 9:
    #         print('task_id[{}], j[{}]'.format(task_id, j))
    #         draw_embeddings = [support_embeddings, query_embeddings]
    #         draw_labels = [support_labels, query_labels]
    #         draw_decision_boundary(draw_embeddings, draw_labels, proto_embeddings, config)

    return loss_sum, query_mi, support_acc, query_acc


def get_FPD_tasks(config):
    FPD_dataset = FPD_process.get_FPD_dataset(config)

    # class 2, 3是随机序列
    # 5-meta-train, 15-meta-test
    # meta_train_class = [i for i in range(7) if i != 3]
    # meta_test_class = [i for i in range(7, 22)] + [3]

    # 5-meta-train, 5-meta-test
    # meta_train_class = [i for i in range(7) if i != 3]
    # meta_test_class = [i for i in range(17, 22)] + [3]

    # 10-meta-train, 10-meta-test
    meta_train_class = [i for i in range(12) if i != 3]
    meta_test_class = [i for i in range(12, 22)] + [3]

    # 10-meta-train, 5-meta-test
    # meta_train_class = [i for i in range(12) if i != 3]
    # meta_test_class = [i for i in range(17, 22)] + [3]

    # 15-meta-train, 5-meta-test
    # meta_train_class = [i for i in range(17) if i != 3]
    # meta_test_class = [i for i in range(17, 22)] + [3]

    # 20-meta-train, 0-meta-test
    # meta_train_class = [i for i in range(22) if i != 3]
    # meta_test_class = [i for i in range(22) if i != 2]

    # 5-meta-train Group 1
    # meta_train_class = [i for i in range(7)] + [2]
    # meta_test_class = [i for i in range(7)] + [3]

    # 5-meta-train Group 2
    # meta_train_class = [i for i in range(7, 12)] + [2]
    # meta_test_class = [i for i in range(7, 12)] + [3]

    # 5-meta-train Group 3
    # meta_train_class = [i for i in range(12, 17)] + [2]
    # meta_test_class = [i for i in range(12, 17)] + [3]

    # 5-meta-train Group 4
    # meta_train_class = [i for i in range(17, 22) if i != 3]
    # meta_test_class = [i for i in range(17, 22) if i != 2]

    print('meta_train_class', meta_train_class)
    print('meta_test_class', meta_test_class)
    FPD_dataset_meta_train = l2l.data.FilteredMetaDataset(FPD_dataset, meta_train_class)
    FPD_dataset_meta_valid = l2l.data.FilteredMetaDataset(FPD_dataset, meta_test_class)
    FPD_dataset_meta_test = l2l.data.FilteredMetaDataset(FPD_dataset, meta_test_class)

    # RemapLabels(FPD_dataset_meta_train, shuffle=True)
    # RemapLabels(FPD_dataset_meta_valid, shuffle=True)
    # RemapLabels(FPD_dataset_meta_test, shuffle=True)

    FPD_train_tasks = l2l.data.TaskDataset(
        dataset=FPD_dataset_meta_train,
        task_transforms=[
            l2l.data.transforms.NWays(FPD_dataset_meta_train, n=config.train_way),
            l2l.data.transforms.KShots(FPD_dataset_meta_train, k=config.train_shot + config.train_query),
            l2l.data.transforms.LoadData(FPD_dataset_meta_train),
        ],
        num_tasks=50000,
    )
    FPD_valid_tasks = l2l.data.TaskDataset(
        dataset=FPD_dataset_meta_valid,
        task_transforms=[
            l2l.data.transforms.NWays(FPD_dataset_meta_valid, n=config.test_way),
            l2l.data.transforms.KShots(FPD_dataset_meta_valid, k=config.test_shot + config.test_query),
            l2l.data.transforms.LoadData(FPD_dataset_meta_valid),
        ],
        num_tasks=50000,
    )
    FPD_test_tasks = l2l.data.TaskDataset(
        dataset=FPD_dataset_meta_test,
        task_transforms=[
            l2l.data.transforms.NWays(FPD_dataset_meta_test, n=config.test_way),
            l2l.data.transforms.KShots(FPD_dataset_meta_test, k=config.test_shot + config.test_query),
            l2l.data.transforms.LoadData(FPD_dataset_meta_test),
        ],
        num_tasks=50000,
    )
    return FPD_train_tasks, FPD_valid_tasks, FPD_test_tasks


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='meta learning config')

    # 项目配置参数
    # parse.add_argument('-learn-name', type=str, default='FPD_ProtoNet_30', help='本次训练名称')
    # parse.add_argument('-learn-name', type=str, default='FPD_MIMML_52', help='本次训练名称')
    parse.add_argument('-learn-name', type=str, default='FPD_MIMML_02', help='本次训练名称')
    parse.add_argument('-path-save', type=str, default='../result/', help='保存字典的位置')
    parse.add_argument('-path-data', type=str, default='../data/task_data/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=False, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0.90, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=0)
    parse.add_argument('-seed', type=int, default=43)
    parse.add_argument('-num_workers', type=int, default=4)

    parse.add_argument('-max-epoch', type=int, default=100)
    parse.add_argument('-meta-batch-size', type=int, default=100)
    parse.add_argument('-lr', type=float, default=0.0001)
    # parse.add_argument('-lr', type=float, default=0.00007)
    # parse.add_argument('-lr', type=float, default=0.00005)
    # parse.add_argument('-lr', type=float, default=0.00002)
    # parse.add_argument('-lr', type=float, default=0.00001)
    parse.add_argument('-reg', type=float, default=0.001)
    # parse.add_argument('-reg', type=float, default=0.0000)

    parse.add_argument('-train-way', type=int, default=5)
    parse.add_argument('-train-shot', type=int, default=5)
    parse.add_argument('-train-query', type=int, default=15)
    parse.add_argument('-test-way', type=int, default=5)
    parse.add_argument('-test-shot', type=int, default=5)
    parse.add_argument('-test-query', type=int, default=15)

    parse.add_argument('-if-MIM', type=bool, default=True)
    # parse.add_argument('-if-MIM', type=bool, default=False)
    parse.add_argument('-train-iteration', type=int, default=1)
    parse.add_argument('-adapt-iteration', type=int, default=10)
    parse.add_argument('-valid-interval', type=int, default=5)
    parse.add_argument('-valid-iteration', type=int, default=200)
    parse.add_argument('-test-iteration', type=int, default=500)

    # parse.add_argument('-if-retrain', type=bool, default=True)
    parse.add_argument('-if-retrain', type=bool, default=False)
    parse.add_argument('-retrain-config', type=str, default=None)
    parse.add_argument('-retrain-model', type=str, default=None)

    # 损失系数
    parse.add_argument('-alpha', type=float, default=0.1)
    parse.add_argument('-lamb', type=float, default=0.1)
    parse.add_argument('-temp', type=float, default=20)
    # parse.add_argument('-if-lr-scheduler', type=bool, default=True)
    parse.add_argument('-if-lr-scheduler', type=bool, default=False)
    parse.add_argument('-lr-step-size', type=int, default=10)
    parse.add_argument('-gamma', type=int, default=0.9)

    # 模型参数配置
    parse.add_argument('-max-len', type=int, default=209 + 2, help='max length of input sequences')
    parse.add_argument('-num-layer', type=int, default=3, help='number of encoder blocks')
    parse.add_argument('-num-head', type=int, default=8, help='number of head in multi-head attention')
    parse.add_argument('-dim-embedding', type=int, default=32, help='residue embedding dimension')
    parse.add_argument('-dim-feedforward', type=int, default=32, help='hidden layer dimension in feedforward layer')
    parse.add_argument('-dim-k', type=int, default=32, help='embedding dimension of vector k or q')
    parse.add_argument('-dim-v', type=int, default=32, help='embedding dimension of vector v')
    parse.add_argument('-vocab-size', type=int, default=28, help='vocab size of word dict')

    config = parse.parse_args()
    print(config)

    # 保存训练配置
    config_dict = config.__dict__
    save_dir = config.path_save + config.learn_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/config.txt', 'w') as f:
        for key, value in config_dict.items():
            key_value_pair = '{}: {}'.format(key, value)
            f.write(key_value_pair + '\r\n')

    random.seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device('cpu')
    if config.cuda and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)

    # load data and construct dataset by learn2learn
    FPD_train_tasks, FPD_valid_tasks, FPD_test_tasks = get_FPD_tasks(config)

    print('len(FPD_train_tasks)', len(FPD_train_tasks), 'type(FPD_train_tasks)', type(FPD_train_tasks))
    print('len(FPD_valid_tasks)', len(FPD_valid_tasks), 'type(FPD_valid_tasks)', type(FPD_valid_tasks))
    print('len(FPD_test_tasks)', len(FPD_test_tasks), 'type(FPD_test_tasks)', type(FPD_test_tasks))

    # Create model
    if config.if_retrain:
        config.retrain_config = '../result/FPD_MIMML_01/config.pkl'
        config_old = pickle.load(open(config.retrain_config, 'rb'))
        config.retrain_model = '../result/FPD_MIMML_01/Epoch[50], Final, ACC[0.5440].pt'
        config_old.train_iteration = 1
        model = Transformer_Encoder.Transformer_Encoder(config_old)
        model = load_model(model, config.retrain_model)
    else:
        model = Transformer_Encoder.Transformer_Encoder(config)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg)

    if config.if_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.gamma)

    print('=' * 100)

    if config.if_lr_scheduler:
        print('current lr = {}'.format(lr_scheduler.get_last_lr()))

    save_dir = config.path_save + config.learn_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/config.pkl', 'wb') as file:
        pickle.dump(config, file)

    # 绘图数据
    step_list = []
    train_loss_list = []
    valid_loss_list = []
    train_accuracy_list = []
    valid_accuracy_list = []

    '''开始训练'''
    pbar = tqdm([i for i in range(1, config.max_epoch + 1)])
    for epoch in pbar:
        optimizer.zero_grad()

        '''Train Episode'''
        model.train()
        train_ctr = 0
        train_loss = 0
        train_acc = 0
        train_mi = 0
        for task_id in range(1, config.meta_batch_size + 1):
            # time_start = time.time()
            train_task = FPD_train_tasks.sample()
            # time_end = time.time()
            # print('FPD_train_tasks time', time_end - time_start)

            if not config.if_MIM:
                # simple version
                loss, acc = fast_adapt(model, train_task, config.train_way, config.train_shot, config.train_query,
                                       device)
                train_ctr += 1
                train_loss += loss.item()
                train_acc += acc.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # MIM version
                for j in range(config.train_iteration):
                    # time_start = time.time()
                    loss_sum, query_mi, support_acc, query_acc = fast_adapt_MIM(model, train_task, config.train_way,
                                                                                config.train_shot, config.train_query,
                                                                                device,
                                                                                config, True)
                    # time_end = time.time()
                    # print('fast_adapt_MIM time', time_end - time_start)
                    optimizer.zero_grad()
                    loss_sum.backward()
                    optimizer.step()

                train_ctr += 1
                train_loss += loss_sum.item()
                train_acc += query_acc.item()
                train_mi += query_mi.item()

        # 调整学习率
        if config.if_lr_scheduler:
            lr_scheduler.step()
            print('current lr = {}'.format(lr_scheduler.get_last_lr()))

        pbar.set_description('epoch[{}]'.format(epoch))
        pbar.set_postfix(accuracy='{0:.4f}'.format(train_acc / train_ctr))
        print()
        '''Train Episode Over'''

        # 记录参数断点
        episodic_state_dict = deepcopy(model.state_dict())  # 必须是deepcopy
        # print('[{}]'.format(epoch), 'episodic parameters:', model.state_dict()['embedding.tok_embed.weight'][0][:8])

        '''Validation Episode'''
        if epoch % config.valid_interval == 0:
            model.eval()
            valid_ctr = 0
            valid_loss = 0
            valid_acc = 0
            valid_mi = 0

            for task_id in range(1, config.valid_iteration + 1):
                # print('\t[{}][{}]'.format(epoch, task_id), 'before reset parameters:',
                #       model.state_dict()['embedding.tok_embed.weight'][0][:8])

                # 每次测试一个任务前都要重新设置模型参数
                model.load_state_dict(episodic_state_dict)

                # print('\t[{}][{}]'.format(epoch, task_id), 'after reset parameters:',
                #       model.state_dict()['embedding.tok_embed.weight'][0][:8])

                valid_task = FPD_valid_tasks.sample()

                if not config.if_MIM:
                    # simple version
                    loss, acc = fast_adapt(model, valid_task, config.test_way, config.test_shot, config.test_query,
                                           device)
                    valid_ctr += 1
                    valid_loss += loss.item()
                    valid_acc += acc.item()
                else:
                    # MIM version
                    for j in range(config.adapt_iteration):
                        loss_sum, query_mi, support_acc, query_acc = fast_adapt_MIM(model, valid_task, config.test_way,
                                                                                    config.test_shot, config.test_query,
                                                                                    device, config, False)
                        optimizer.zero_grad()
                        loss_sum.backward()
                        optimizer.step()

                        # 查看每一次adaptation的情况
                        # print('\t\t[{}][{}] | valid | valid_loss={:.4f} | valid_acc={:.4f} | valid_mi={:.4f}'
                        #       .format(epoch, task_id, loss_sum.item(), query_acc.item(), query_mi.item()))

                    valid_ctr += 1
                    valid_loss += loss_sum.item()
                    valid_acc += query_acc.item()
                    valid_mi += query_mi.item()

                if task_id % 1 == 0:
                    print('\tepoch[{}] | valid[{}] | valid_loss={:.4f} | valid_acc={:.4f} | valid_mi={:.4f}'
                          .format(epoch, task_id, valid_loss / valid_ctr, valid_acc / valid_ctr, valid_mi / valid_ctr))

            if valid_ctr != 0:
                print('\tepoch[{}] | valid final | valid_loss={:.4f} | valid_acc={:.4f} | valid_mi={:.4f} '
                      .format(epoch, valid_loss / valid_ctr, valid_acc / valid_ctr, valid_mi / valid_ctr))

            step_list.append(epoch)
            train_loss_list.append(train_loss / train_ctr)
            train_accuracy_list.append(train_acc / train_ctr)
            valid_loss_list.append(valid_loss / valid_ctr if valid_ctr != 0 else 0)
            valid_accuracy_list.append(valid_acc / valid_ctr if valid_ctr != 0 else 0)

            # 绘图
            fig_name = 'Epoch[{}]'.format(epoch)
            draw_figure(fig_name)

            # 保存模型参数
            if valid_ctr != 0:
                save_path_pt = save_model(episodic_state_dict, valid_acc / valid_ctr,
                                          config.path_save + config.learn_name,
                                          'Epoch[{}]'.format(epoch))

            model.load_state_dict(episodic_state_dict)  # 切换为原来训练时候的参数

        print('-' * 100)
        '''Valid Episode Over'''

    '''Test Period'''
    time_start = time.time()

    # print('[{}]'.format('0000'), 'before test parameters:',
    #       model.state_dict()['embedding.tok_embed.weight'][0][:8])

    final_state_dict = deepcopy(model.state_dict())
    model.load_state_dict(final_state_dict)

    # print('[{}]'.format('0000'), 'ready test parameters:',
    #       model.state_dict()['embedding.tok_embed.weight'][0][:8])

    model.eval()
    test_ctr = 0
    test_loss = 0
    test_acc = 0
    test_mi = 0
    for task_id in range(1, config.test_iteration + 1):
        # 每次测试一个任务前都要重新设置模型参数
        model.load_state_dict(final_state_dict)

        test_task = FPD_test_tasks.sample()

        if not config.if_MIM:
            # simple version
            loss, acc = fast_adapt(model, test_task, config.test_way, config.test_shot, config.test_query, device)
            test_ctr += 1
            test_loss += loss.item()
            test_acc += acc.item()
        else:
            # MIM version
            for j in range(config.adapt_iteration):
                loss_sum, query_mi, support_acc, query_acc = fast_adapt_MIM(model, test_task, config.test_way,
                                                                            config.test_shot, config.test_query,
                                                                            device, config, False)
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
            test_ctr += 1
            test_loss += loss_sum.item()
            test_acc += query_acc.item()
            test_mi += query_mi.item()

        print('task_id[{}] | test_loss={:.4f} | test_acc={:.4f} | test_mi={:.4f}'
              .format(task_id, test_loss / test_ctr, test_acc / test_ctr, test_mi / test_ctr))

        if task_id >= len(FPD_test_tasks):
            break

    time_end = time.time()
    print('Model Test time', time_end - time_start)

    save_model(final_state_dict, test_acc / test_ctr, config.path_save + config.learn_name,
               'Epoch[{}], Final'.format(config.max_epoch))

