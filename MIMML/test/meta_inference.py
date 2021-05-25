import torch
import pickle
import time
import numpy as np
import torch.nn.functional as F
import learn2learn as l2l

from preprocess import FPD_process
from copy import deepcopy
from model import Transformer_Encoder
from train.model_operation import save_model, load_model


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

        '''L_S_CE + L_Q_MI'''
        # loss_sum = config.lamb * loss_support_CE - (query_ent - config.alpha * query_cond_ent)

    else:
        '''L_S_CE + L_S_MI + L_Q_MI'''
        loss_sum = config.lamb * (loss_support_CE) - \
                   (support_ent - config.alpha * support_cond_ent) - \
                   (query_ent - config.alpha * query_cond_ent)

        '''L_S_CE + L_Q_MI'''
        # loss_sum = config.lamb * (loss_support_CE) + \
        #            (query_ent - config.alpha * query_cond_ent)

        '''L_S_CE + L_Q_MI'''
        # loss_sum = config.lamb * loss_support_CE - (query_ent - config.alpha * query_cond_ent)

    return loss_sum, query_mi, support_acc, query_acc


def get_FPD_tasks(config):
    FPD_dataset = FPD_process.get_FPD_dataset(config)

    # class 2, 3是随机序列
    # 5-meta-train, 15-meta-test
    # meta_train_class = [i for i in range(7) if i != 3]
    # meta_test_class = [i for i in range(7, 22)] + [3]

    # 10-meta-train, 10-meta-test
    # meta_train_class = [i for i in range(12) if i != 3]
    # meta_test_class = [i for i in range(12, 22)] + [3]

    # 15-meta-train, 5-meta-test
    meta_train_class = [i for i in range(17) if i != 3]
    meta_test_class = [i for i in range(17, 22)] + [3]

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
        num_tasks=50000,
    )
    return FPD_train_tasks, FPD_valid_tasks, FPD_test_tasks


if __name__ == '__main__':
    # config_path = '../result/FPD_MIMML_00/config.pkl'
    # model_path = '../result/FPD_MIMML_00/Epoch[40], Final, ACC[0.6243].pt'

    # config_path = '../result/std_model/Train[10], Test[10], Epoch[30], ACC[0.5420].pkl'
    # model_path = '../result/std_model/Train[10], Test[10], Epoch[30], ACC[0.5420].pt'

    config_path = '../result/FPD_MIMML_00/config.pkl'
    model_path = '../result/FPD_MIMML_00/Epoch[50], ACC[0.6335].pt'

    config = pickle.load(open(config_path, 'rb'))
    device = torch.device('cpu')
    if config.cuda and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)

    print('token2index', config.token2index)

    # config.train_way = 2
    # config.train_shot = 5
    # config.train_query = 5
    #
    # config.test_way = 2
    # config.test_shot = 5
    # config.test_query = 200
    #
    # config.test_iteration = 10
    config.adapt_iteration = 10

    config.if_lr_scheduler = False
    # config.if_lr_scheduler = True
    config.lr = 0.00005
    config.lr_step_size = 50
    config.gamma = 0.9

    FPD_train_tasks, FPD_valid_tasks, FPD_test_tasks = get_FPD_tasks(config)

    model = Transformer_Encoder.Transformer_Encoder(config)
    model.to(device)

    print('before load', model.state_dict()['embedding.tok_embed.weight'][0])
    model = load_model(model, model_path)
    print('after load', model.state_dict()['embedding.tok_embed.weight'][0])

    '''Test Period'''
    time_start = time.time()

    final_state_dict = deepcopy(model.state_dict())

    test_ctr = 0
    test_loss = 0
    test_acc = 0
    test_mi = 0
    for task_id in range(1, config.test_iteration + 1):
        # model.train()
        model.eval()

        # 每次测试一个任务前都要重新设置模型参数
        model.load_state_dict(final_state_dict)

        # test_task = FPD_test_tasks.sample()
        # test_task = FPD_train_tasks.sample()
        test_task = FPD_valid_tasks.sample()

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.reg)
        if config.if_lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.lr_step_size, gamma=config.gamma)
            print('current start lr:', lr_scheduler.get_last_lr())

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

            if config.if_lr_scheduler:
                lr_scheduler.step()

        print('task_id[{}] | test_loss={:.4f} | test_acc={:.4f} | test_mi={:.4f}'
              .format(task_id, test_loss / test_ctr, test_acc / test_ctr, test_mi / test_ctr))

    time_end = time.time()
    print('Model Test time', time_end - time_start)
