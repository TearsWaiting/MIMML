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


def get_FPD_tasks(config):
    FPD_dataset = FPD_process.get_FPD_dataset(config)

    # 2, 3是随机序列
    meta_train_class = [i for i in range(12) if i != 3]
    meta_test_class = [i for i in range(12, 22)] + [3]

    # meta_train_class = [i for i in range(13) if i != 2 and i != 3 and i != 4]
    # meta_test_class = [i for i in range(13, 22)]
    print('meta_train_class', meta_train_class)
    print('meta_test_class', meta_test_class)
    FPD_dataset_meta_train = l2l.data.FilteredMetaDataset(FPD_dataset, meta_train_class)
    FPD_dataset_meta_valid = l2l.data.FilteredMetaDataset(FPD_dataset, meta_test_class)
    FPD_dataset_meta_test = l2l.data.FilteredMetaDataset(FPD_dataset, meta_test_class)

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
        num_tasks=1000,
    )
    return FPD_train_tasks, FPD_valid_tasks, FPD_test_tasks


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


def fast_adapt_MIM(model, task, way, shot, query_num, device, config, if_train):
    # print('if_train', if_train)
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
    support_probs = support_logits.softmax(1)
    query_probs = query_logits.softmax(1)

    # 计算 Entropy, Conditional Entropy and Mutual Information
    support_ent = get_entropy(probs=support_probs.detach())
    support_cond_ent = get_cond_entropy(probs=support_probs.detach())
    # support_mi = support_ent - support_cond_ent

    query_ent = get_entropy(probs=query_probs.detach())
    query_cond_ent = get_cond_entropy(probs=query_probs.detach())
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
    else:
        '''L_S_CE + L_S_MI + L_Q_MI'''
        loss_sum = config.lamb * (loss_support_CE) - \
                   (support_ent - config.alpha * support_cond_ent) - \
                   (query_ent - config.alpha * query_cond_ent)

        '''L_S_CE + L_Q_MI'''
        # loss_sum = config.lamb * (loss_support_CE) + \
        #            (query_ent - config.alpha * query_cond_ent)

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
    parse.add_argument('-meta-batch-size', type=int, default=100)
    # parse.add_argument('-lr', type=float, default=0.00003)
    parse.add_argument('-lr', type=float, default=0.0001)
    parse.add_argument('-train-way', type=int, default=5)
    parse.add_argument('-train-shot', type=int, default=5)
    parse.add_argument('-train-query', type=int, default=15)
    parse.add_argument('-test-way', type=int, default=5)
    parse.add_argument('-test-shot', type=int, default=5)
    parse.add_argument('-test-query', type=int, default=15)
    parse.add_argument('-num-valid-iter', type=int, default=15)
    parse.add_argument('-num-mi-iter', type=int, default=15)

    # 损失系数
    parse.add_argument('-alpha', type=float, default=0.1)
    parse.add_argument('-lamb', type=float, default=0.1)
    parse.add_argument('-temp', type=float, default=20)

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

    # load data and construct dataset by learn2learn
    FPD_train_tasks, FPD_valid_tasks, FPD_test_tasks = get_FPD_tasks(config)

    print('len(FPD_train_tasks)', len(FPD_train_tasks), 'type(FPD_train_tasks)', type(FPD_train_tasks))
    print('len(FPD_valid_tasks)', len(FPD_valid_tasks), 'type(FPD_valid_tasks)', type(FPD_valid_tasks))
    print('len(FPD_test_tasks)', len(FPD_test_tasks), 'type(FPD_test_tasks)', type(FPD_test_tasks))

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
    pbar = tqdm([i for i in range(1, config.max_epoch)])
    for epoch in pbar:
        optimizer.zero_grad()

        '''Train Period'''
        model.train()
        train_ctr = 0
        train_loss = 0
        train_acc = 0
        train_mi = 0
        for task_id in range(1, config.meta_batch_size):
            # time_start = time.time()
            train_task = FPD_train_tasks.sample()
            # time_end = time.time()
            # print('FPD_train_tasks time', time_end - time_start)

            # simple version
            loss, acc = fast_adapt(model, train_task, config.train_way, config.train_shot, config.train_query, device)
            train_ctr += 1
            train_loss += loss.item()
            train_acc += acc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # MIM version
            # time_start = time.time()
            # loss_sum, query_mi, support_acc, query_acc = fast_adapt_MIM(model, train_task, config.train_way,
            #                                                             config.train_shot, config.train_query, device,
            #                                                             config, True)
            # time_end = time.time()
            # print('fast_adapt_MIM time', time_end - time_start)
            # train_ctr += 1
            # train_loss += loss_sum.item()
            # train_acc += query_acc.item()
            # train_mi += query_mi.item()
            # optimizer.zero_grad()
            # loss_sum.backward()
            # optimizer.step()

        # lr_scheduler.step()

        '''Validation Period'''
        model.eval()
        valid_ctr = 0
        valid_loss = 0
        valid_acc = 0
        valid_mi = 0
        for task_id in range(config.num_valid_iter):
            valid_task = FPD_valid_tasks.sample()

            # simple version
            loss, acc = fast_adapt(model, valid_task, config.test_way, config.test_shot, config.test_query, device)
            valid_ctr += 1
            valid_loss += loss.item()
            valid_acc += acc.item()

            # MIM version
            # for j in range(config.num_mi_iter):
            #     loss_sum, query_mi, support_acc, query_acc = fast_adapt_MIM(model, valid_task, config.test_way,
            #                                                                 config.test_shot, config.test_query,
            #                                                                 device, config, False)
            #     optimizer.zero_grad()
            #     loss_sum.backward()
            #     optimizer.step()
            # valid_ctr += 1
            # valid_loss += loss_sum.item()
            # valid_acc += query_acc.item()
            # valid_mi += query_mi.item()

        print()
        train_loss = train_loss / train_ctr
        train_acc = train_acc / train_ctr
        train_mi = train_mi / train_ctr
        valid_loss = valid_loss / valid_ctr
        valid_acc = valid_acc / valid_ctr
        valid_mi = valid_mi / valid_ctr
        print('epoch[{}] | train_loss={:.4f} | train_acc={:.4f} | train_mi={:.4f} '
              '| valid_loss={:.4f} | valid_acc={:.4f} | valid_mi={:.4f}'
              .format(epoch, train_loss, train_acc, train_mi, valid_loss, valid_acc, valid_mi))

        pbar.set_description('epoch[{}]'.format(epoch))
        pbar.set_postfix(accuracy='{0:.4f}'.format(valid_acc))

        if epoch % 1 == 0:
            step_list.append(epoch)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            train_accuracy_list.append(train_acc)
            valid_accuracy_list.append(valid_acc)

        if epoch % 10 == 0 and epoch != 1:
            draw_figure()

    '''Test Period'''
    test_ctr = 0
    test_loss = 0
    test_acc = 0
    test_mi = 0
    for task_id, task_batch_test in enumerate(FPD_test_tasks, 1):
        test_task = FPD_valid_tasks.sample()

        # simple version
        test_loss, test_acc = fast_adapt(model, test_task, config.test_way, config.test_shot, config.test_query, device)

        # MIM version
        # for j in range(config.num_mi_iter):
        #     loss_sum, query_mi, support_acc, query_acc = fast_adapt_MIM(model, test_task, config.test_way,
        #                                                                 config.test_shot, config.test_query,
        #                                                                 device, config, False)
        #     optimizer.zero_grad()
        #     loss_sum.backward()
        #     optimizer.step()
        # test_ctr += 1
        # test_loss += loss_sum.item()
        # test_acc += query_acc.item()
        # test_mi += query_mi.item()

        print('task_id[{}] | test_loss={:.4f} | test_acc={:.4f} | test_mi={:.4f}'
              .format(task_id, test_loss / test_ctr, test_acc / test_ctr, test_mi / test_ctr))

        if task_id >= len(FPD_test_tasks):
            break
