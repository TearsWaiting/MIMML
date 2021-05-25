import torch
import pickle
import time
import numpy as np
import torch.nn.functional as F
import learn2learn as l2l

from preprocess import FPD_process, data_loader
from copy import deepcopy
from model import Transformer_Encoder, Transformer_Encoder_finetune
from train.model_operation import save_model, load_model
from util import util_metric


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


def fast_adapt_MIM(model, task, way, shot, query_num, device, config, if_train):
    data, labels = task

    # print('-' * 100)
    # print('data_num', len(data))
    # print('labels_num', len(labels))

    # 这里无需转换，原因未知
    # data = [x.view(-1, 1) for x in data]
    # data = torch.cat(data, dim=1)
    # data = data.to(device)

    # print('data', data.size(), data)
    # print('label', labels.size(), labels)

    # 无需reset，二分类
    # print('labels', labels.size(), labels)
    reset_labels = np.repeat(np.arange(way), shot + query_num, 0)
    reset_labels = torch.from_numpy(reset_labels)
    labels = reset_labels.to(device)
    # print('labels', labels.size(), labels)

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    # '''检查'''
    # print('data_num', len(data))
    # print('labels_num', len(labels))

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
        # loss_sum = config.lamb * loss_support_CE - (query_ent - config.alpha * query_cond_ent)

    # 计算 ACC
    # support_acc = accuracy(support_logits, support_labels)
    # query_acc = accuracy(query_logits, query_labels)
    # print('{}: {:4f} | {}: {:4f}'.format('support_acc', support_acc, 'query_acc', query_acc))

    # 计算 metric
    query_pred_label = query_logits.argmax(dim=1).view(query_labels.shape).detach().cpu()
    query_labels = query_labels.detach().cpu()
    query_probs = query_probs[:, 1]
    query_probs = query_probs.detach().cpu()
    # print('query_pred_label', query_pred_label.size())
    # print('query_labels', query_labels.size())
    # print('query_probs', query_probs.size())
    metric, roc_data, prc_data = util_metric.caculate_metric(query_pred_label, query_labels, query_probs)

    return loss_sum, query_mi, metric, roc_data, prc_data


def direct_SL(model, task, way, shot, query_num, device, config, if_train):
    data, labels = task

    reset_labels = np.repeat(np.arange(way), shot + query_num, 0)
    reset_labels = torch.from_numpy(reset_labels)
    labels = reset_labels.to(device)

    # Sort data samples by labels
    sort = torch.sort(labels)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)

    logits = model(data)[0]
    support_indices = np.zeros(data.size(0), dtype=bool)
    selection = np.arange(way) * (shot + query_num)
    for offset in range(shot):
        support_indices[selection + offset] = True
    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)

    support_logits = logits[support_indices]
    query_logits = logits[query_indices]
    support_labels = labels[support_indices].long()
    query_labels = labels[query_indices].long()

    loss_support_CE = (F.cross_entropy(support_logits, support_labels).float()).mean()

    if if_train:
        loss_query_CE = (F.cross_entropy(query_logits, query_labels).float()).mean()

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

    if if_train:
        loss_sum = loss_support_CE + loss_query_CE
    else:
        loss_sum = loss_support_CE

    # 计算 metric
    query_pred_label = query_logits.argmax(dim=1).view(query_labels.shape).detach().cpu()
    query_labels = query_labels.detach().cpu()
    query_probs = query_probs[:, 1]
    query_probs = query_probs.detach().cpu()
    metric, roc_data, prc_data = util_metric.caculate_metric(query_pred_label, query_labels, query_probs)

    return loss_sum, query_mi, metric, roc_data, prc_data
    # return loss_sum, torch.tensor([0]), metric, roc_data, prc_data


def get_finetune_tasks(config):
    finetune_dataset = data_loader.get_finetune_dataset(config)
    finetune_meta_dataset = l2l.data.MetaDataset(finetune_dataset)

    # 没有越界
    # for i, sample in enumerate(finetune_meta_dataset):
    #     print('sample[{}]:{}'.format(i, sample))
    #     for j in sample[0]:
    #         j = int(j)
    #         if j < 0 or j >= 28:
    #             print('error!')

    for i in range(5):
        print('finetune_meta_dataset[{}]:{}'.format(i, finetune_meta_dataset[i]))
    print('-' * 100)

    finetune_tasks = l2l.data.TaskDataset(
        dataset=finetune_meta_dataset,
        task_transforms=[
            l2l.data.transforms.NWays(finetune_meta_dataset, n=config.test_way),
            l2l.data.transforms.KShots(finetune_meta_dataset, k=config.test_shot + config.test_query),
            l2l.data.transforms.LoadData(finetune_meta_dataset),
        ],
        num_tasks=1000,
    )
    return finetune_tasks


def select_dataset():
    # finetune dataset
    # Functional Peptides finetune dataset
    # path_train_data = '../data/task_data/finetune dataset/Anti-angiogenic Peptides/benchmarkdataset.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-angiogenic Peptides/NT15dataset.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-cancer Peptides/ACP2_main_train.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-cancer Peptides/ACP2_main_test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-cancer Peptides/ACP2_alternate_train.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Anti-cancer Peptides/ACP2_alternate_test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Anti-fungal Peptides/AntiFP_main.tsv'
    # path_test_data = None
    path_train_data = '../data/task_data/finetune dataset/Anti-fungal Peptides/Training_antifngl_DS1.tsv'
    path_test_data = None
    # path_train_data = '../data/task_data/finetune dataset/Anti-fungal Peptides/independent_main1.tsv'
    # path_test_data = None
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
    # path_train_data = '../data/task_data/finetune dataset/Cell-penetrating Peptides/training.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Cell-penetrating Peptides/test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/DPP IV Inhibitor/training.tsv'
    # path_test_data = '../data/task_data/finetune dataset/DPP IV Inhibitor/test.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Neuropeptides/training_ds1.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Neuropeptides/test_ds1.tsv'
    # path_train_data = '../data/task_data/finetune dataset/Neuropeptides/training_ds2.tsv'
    # path_test_data = '../data/task_data/finetune dataset/Neuropeptides/test_ds2.tsv'
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
    #
    return path_train_data, path_test_data


if __name__ == '__main__':
    # config_path = '../result/Shot[5], Train[15], Test[5], Epoch[70], 00/config.pkl'
    # model_path = '../result/Shot[5], Train[15], Test[5], Epoch[70], 00/Epoch[50], ACC[0.6611].pt'

    # config_path = '../result/FPD_MIMML_5_G1/config.pkl'
    # model_path = '../result/FPD_MIMML_5_G1/Epoch[50], Final, ACC[0.6559].pt'

    # config_path = '../result/FPD_MIMML_5_G2/config.pkl'
    # model_path = '../result/FPD_MIMML_5_G2/Epoch[50], Final, ACC[0.7467].pt'

    # config_path = '../result/FPD_MIMML_5_G3/config.pkl'
    # model_path = '../result/FPD_MIMML_5_G3/Epoch[50], Final, ACC[0.6737].pt'
    # model_path = '../result/FPD_MIMML_5_G4/Epoch[15], ACC[0.5945].pt'

    config_path = '../result/FPD_MIMML_5_G4/config.pkl'
    model_path = '../result/FPD_MIMML_5_G4/Epoch[50], Final, ACC[0.9960].pt'
    # model_path = '../result/FPD_MIMML_5_G4/Epoch[15], ACC[0.8820].pt'

    config = pickle.load(open(config_path, 'rb'))
    device = torch.device('cpu')
    config.device = 0
    if config.cuda and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)

    print('token2index', config.token2index)

    config.if_MIM = True
    # config.if_MIM = False

    config.test_way = 2
    # config.test_shot = 27
    # config.test_query = 133

    config.test_shot = int(934 * 0.05)
    # config.test_shot = 12
    config.test_query = 233

    config.test_iteration = 20
    config.adapt_iteration = 300

    config.if_lr_scheduler = False
    # config.if_lr_scheduler = True
    # config.lr = 0.00025
    # config.lr = 0.0001
    config.lr = 0.00005  # AIP
    config.lr_step_size = 50
    config.gamma = 0.95

    print('config', config)

    path_train_data, path_test_data = select_dataset()
    config.path_train_data = path_train_data
    config.path_test_data = path_test_data
    fine_tune_tasks = get_finetune_tasks(config)

    if config.if_MIM:
        model = Transformer_Encoder.Transformer_Encoder(config)
    else:
        model = Transformer_Encoder_finetune.Transformer_Encoder(config)
    model.to(device)

    print('before load', model.state_dict()['embedding.tok_embed.weight'][0])
    if config.if_MIM:
        model = load_model(model, model_path)
    print('after load', model.state_dict()['embedding.tok_embed.weight'][0])

    '''Test Period'''
    time_start = time.time()

    final_state_dict = deepcopy(model.state_dict())

    test_ctr = 0
    test_loss = 0
    test_mi = 0
    test_acc = 0
    test_se = 0
    test_sp = 0
    test_mcc = 0
    test_auc = 0
    for task_id in range(1, config.test_iteration + 1):
        # model.train()
        model.eval()

        # 每次测试一个任务前都要重新设置模型参数, 已验证有效无误
        # print('before reset', model.state_dict()['embedding.tok_embed.weight'][0])
        model.load_state_dict(final_state_dict)
        # print('after reset', model.state_dict()['embedding.tok_embed.weight'][0])

        test_task = fine_tune_tasks.sample()

        # 对于每个任务都重新设置一下比较稳妥
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.reg)
        if config.if_lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.lr_step_size, gamma=config.gamma)

            # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)

            # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
            #                                                    total_steps=config.adapt_iteration)

            print('current start lr:', lr_scheduler.get_last_lr())

        # 记录该测试任务的最佳性能
        best_loss = 0
        best_mi = 0
        best_acc = 0
        best_se = 0
        best_sp = 0
        best_mcc = 0
        best_auc = 0

        if config.if_MIM:
            # MIM version
            for j in range(config.adapt_iteration):
                loss_sum, query_mi, metric, roc_data, prc_data = fast_adapt_MIM(model, test_task, config.test_way,
                                                                                config.test_shot, config.test_query,
                                                                                device, config, False)
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()

                if metric[0] > best_acc:
                    best_loss = loss_sum
                    best_mi = query_mi
                    best_acc = metric[0]
                    best_se = metric[2]
                    best_sp = metric[3]
                    best_mcc = metric[6]
                    best_auc = roc_data[2]

                if config.if_lr_scheduler:
                    lr_scheduler.step()
                    print('\t\t[{}]: ACC: {:4f} | query_mi: {:4f} | lr: {}'.format(j, metric[0], query_mi.item(),
                                                                                   lr_scheduler.get_last_lr()))
                else:
                    print('\t\t[{}]: ACC: {:4f} | query_mi: {:4f} | lr: {}'.format(j, metric[0], query_mi.item(),
                                                                                   config.lr))

            if config.if_lr_scheduler:
                print('current end lr:', lr_scheduler.get_last_lr())

            test_ctr += 1
            test_loss += best_loss.item()
            test_mi += best_mi.item()
            test_acc += best_acc.item()
            test_se += best_se.item()
            test_sp += best_sp.item()
            test_mcc += best_mcc.item()
            test_auc += best_auc.item()

            # metric: [ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC]
            # roc_data: [fpr, tpr, AUC]
            # prc_data: [recall, precision, AP]
        else:
            # direct supervised learning
            for j in range(config.adapt_iteration):
                loss_sum, query_mi, metric, roc_data, prc_data = direct_SL(model, test_task, config.test_way,
                                                                           config.test_shot, config.test_query,
                                                                           device, config, False)
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()

                if metric[0] > best_acc:
                    best_loss = loss_sum
                    best_mi = query_mi
                    best_acc = metric[0]
                    best_se = metric[2]
                    best_sp = metric[3]
                    best_mcc = metric[6]
                    best_auc = roc_data[2]

                if config.if_lr_scheduler:
                    lr_scheduler.step()
                    print('\t\t[{}]: ACC: {:4f} | query_mi: {:4f} | lr: {}'.format(j, metric[0], query_mi.item(),
                                                                                   lr_scheduler.get_last_lr()))
                else:
                    print('\t\t[{}]: ACC: {:4f} | query_mi: {:4f} | lr: {}'.format(j, metric[0], query_mi.item(),
                                                                                   config.lr))

            test_ctr += 1
            test_loss += best_loss.item()
            test_mi += best_mi.item()
            test_acc += best_acc.item()
            test_se += best_se.item()
            test_sp += best_sp.item()
            test_mcc += best_mcc.item()
            test_auc += best_auc.item()

        print('task_id[{}] | test_loss={:.4f} | test_mi={:.4f} | '
              'test_acc={:.4f} | test_se={:.4f} | test_sp={:.4f} | test_mcc={:.4f} | test_auc={:.4f}'
              .format(task_id, test_loss / test_ctr, test_mi / test_ctr, test_acc / test_ctr,
                      test_se / test_ctr, test_sp / test_ctr, test_mcc / test_ctr, test_auc / test_ctr))

    time_end = time.time()
    print('Model Test time', time_end - time_start)
