# ---encoding:utf-8---
# @Time : 2021.03.30
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : FSL_pretrain.py


import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import learn2learn as l2l
import numpy as np
import torch.utils.data as Data

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import miniimagenet
from preprocess import FPD_process
from model import Transformer_Encoder
from tqdm import tqdm


def draw_figure():
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

    # plt.savefig(config.result_folder + '/' + fig_name + '.png')
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


def get_FPD(config):
    FPD_dataset = FPD_process.get_FPD_dataset(config)
    data_loader = Data.DataLoader(FPD_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    return data_loader


def fast_adapt(model, task, way, shot, query_num, device):
    data, labels = task
    data = [x.view(-1, 1) for x in data]
    data = torch.cat(data, dim=1)
    data = data.to(device)

    # print('labels', labels.size(), labels)
    reset_labels = np.repeat(np.arange(way), shot + query_num, 0)
    reset_labels = torch.from_numpy(reset_labels)
    labels = reset_labels.to(device)
    # print('labels', labels.size(), labels)

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
    support = embeddings[support_indices]
    support = support.reshape(way, shot, -1).mean(dim=1)
    query = embeddings[query_indices]
    labels = labels[query_indices].long()

    logits = pairwise_distances_logits(query, support, 1)
    loss = F.cross_entropy(logits, labels)
    acc = accuracy(logits, labels)
    return loss, acc


def fast_adapt_MIM(model, task, way, shot, query_num, device, config, if_train=True):
    data, labels = task
    data = [x.view(-1, 1) for x in data]
    data = torch.cat(data, dim=1)
    data = data.to(device)

    # print('labels', labels.size(), labels)
    reset_labels = np.repeat(np.arange(way), shot + query_num, 0)
    reset_labels = torch.from_numpy(reset_labels)
    labels = reset_labels.to(device)
    # print('labels', labels.size(), labels)

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

    # 计算 embeddings
    support_embeddings = embeddings[support_indices]
    # support_embeddings: [shot, dim_feature]
    proto_embeddings = support_embeddings.reshape(way, shot, -1).mean(dim=1)
    # proto_embeddings: [way, dim_feature]
    query_embeddings = embeddings[query_indices]
    # support: [query_num, dim_feature]

    # 对embedding进行归一化
    support_embeddings = F.normalize(support_embeddings.view(support_embeddings.size(0), -1), dim=1)
    proto_embeddings = F.normalize(proto_embeddings.view(proto_embeddings.size(0), -1), dim=1)
    query_embeddings = F.normalize(query_embeddings.view(query_embeddings.size(0), -1), dim=1)

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
    # support_probs = support_logits.softmax(1)
    query_probs = query_logits.softmax(1)

    # 计算 Entropy, Conditional Entropy and Mutual Information
    # get_MI(probs=query_probs)
    query_ent = get_entropy(probs=query_probs.detach())
    query_cond_ent = get_cond_entropy(probs=query_probs.detach())
    query_mi = query_ent - query_cond_ent

    # 计算总损失
    if if_train:
        loss_sum = config.lamb * (loss_support_CE + loss_query_CE) + query_ent - config.alpha * query_cond_ent
    else:
        loss_sum = config.lamb * (loss_support_CE) + query_ent - config.alpha * query_cond_ent

    return loss_sum, query_mi, support_acc, query_acc


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='meta learning config')

    # 项目配置参数
    parse.add_argument('-learn-name', type=str, default='FPD_pretrain_0', help='本次训练名称')
    parse.add_argument('-path-save', type=str, default='../result/', help='保存字典的位置')
    parse.add_argument('-path-data', type=str, default='../data/task_data/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=False, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0.90, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=1)
    parse.add_argument('-seed', type=int, default=43)
    parse.add_argument('-num_workers', type=int, default=4)

    parse.add_argument('-max-epoch', type=int, default=200)
    parse.add_argument('-batch-size', type=int, default=32)
    parse.add_argument('-lr', type=float, default=0.00003)
    parse.add_argument('-train-way', type=int, default=17)
    # parse.add_argument('-train-shot', type=int, default=5)
    # parse.add_argument('-train-query', type=int, default=15)
    # parse.add_argument('-test-way', type=int, default=5)
    # parse.add_argument('-test-shot', type=int, default=5)
    # parse.add_argument('-test-query', type=int, default=15)
    # parse.add_argument('-num-valid-iter', type=int, default=15)
    # parse.add_argument('-num-mi-iter', type=int, default=15)

    # 损失系数
    # parse.add_argument('-alpha', type=float, default=0.1)
    # parse.add_argument('-lamb', type=float, default=0.1)
    # parse.add_argument('-temp', type=float, default=10)

    # 模型参数配置
    parse.add_argument('-max-len', type=int, default=209 + 2, help='max length of input sequences')
    parse.add_argument('-num-layer', type=int, default=1, help='number of encoder blocks')
    parse.add_argument('-num-head', type=int, default=8, help='number of head in multi-head attention')
    parse.add_argument('-dim-embedding', type=int, default=32, help='residue embedding dimension')
    parse.add_argument('-dim-feedforward', type=int, default=32, help='hidden layer dimension in feedforward layer')
    parse.add_argument('-dim-k', type=int, default=32, help='embedding dimension of vector k or q')
    parse.add_argument('-dim-v', type=int, default=32, help='embedding dimension of vector v')
    parse.add_argument('-vocab-size', type=int, default=28, help='vocab size of word dict')

    config = parse.parse_args()
    print(config)

    device = torch.device('cpu')
    if config.cuda and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)

    # load all FPD data
    data_loader = get_FPD(config)
    print('len(data_loader)', len(data_loader))

    # Create model
    model = Transformer_Encoder.Transformer_Encoder(config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=20, gamma=0.8)

    # 绘图数据
    step_list = []
    train_loss_list = []
    valid_loss_list = []
    train_accuracy_list = []
    valid_accuracy_list = []

    '''开始训练'''
    pbar = tqdm([i for i in range(config.max_epoch)])
    for epoch in pbar:
        optimizer.zero_grad()

        '''Train Period'''
        model.train()
        train_ctr = 0
        train_loss = 0
        train_acc = 0
        train_mi = 0

        # TODO 补完预训练
