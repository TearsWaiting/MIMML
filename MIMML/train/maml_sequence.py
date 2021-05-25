# ---encoding:utf-8---
# @Time : 2021.03.07
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : maml_sequence.py


import random
import numpy as np
import argparse
import torch
import learn2learn as l2l
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch import nn, optim
from preprocess import FPD_process
from model import BERT


def draw_figure():
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

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


def get_accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(task, learner, loss, adaptation_steps):
    support_samples, support_labels, query_samples, query_labels = task

    # Adapt the model
    for step in range(adaptation_steps):
        predictions = learner(support_samples)[0]
        adaptation_error = loss(predictions, support_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(query_samples)[0]
    evaluation_error = loss(predictions, query_labels)
    evaluation_accuracy = get_accuracy(predictions, query_labels)
    return evaluation_error, evaluation_accuracy


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
    # meta_train_class = [i for i in range(12) if i != 3]
    # meta_test_class = [i for i in range(12, 22)] + [3]

    # 10-meta-train, 5-meta-test
    meta_train_class = [i for i in range(12) if i != 3]
    meta_test_class = [i for i in range(17, 22)] + [3]

    # 15-meta-train, 5-meta-test
    # meta_train_class = [i for i in range(17) if i != 3]
    # meta_test_class = [i for i in range(17, 22)] + [3]

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


def get_task_data(task_set, way, shot, query):
    task_data = task_set.sample()
    task_samples = task_data[0]
    task_labels = task_data[1]

    # print('task_samples', len(task_samples)) # task_samples: max_len = 209
    # print('task_labels', task_labels.size(), task_labels)  # task_labels: [ways * (shots + test_shots)]

    # 对输入序列进行维度变换（必需的）
    task_samples = [x.view(-1, 1) for x in task_samples]
    task_samples = torch.cat(task_samples, dim=1)
    # task_samples: [num_samples = shots + test_shots = 5 + 15 = 20, max_len = 209]

    # 划分support set 和 query set
    support_indices = np.zeros(task_samples.size(0), dtype=bool)
    indices = np.array([i for i in range(len(task_labels)) if i % (shot + query) < shot])
    support_indices[indices] = True

    query_indices = torch.from_numpy(~support_indices)
    support_indices = torch.from_numpy(support_indices)

    train_support_samples, train_support_labels = task_samples[support_indices], task_labels[support_indices]
    train_query_samples, train_query_labels = task_samples[query_indices], task_labels[query_indices]

    # 标签映射
    label_reset_support = np.repeat(np.arange(way), shot, 0)
    label_reset_query = np.repeat(np.arange(way), query, 0)
    train_support_labels = torch.from_numpy(label_reset_support)
    train_query_labels = torch.from_numpy(label_reset_query)

    train_support_samples, train_support_labels = train_support_samples.to(device), train_support_labels.to(device)
    train_query_samples, train_query_labels = train_query_samples.to(device), train_query_labels.to(device)

    return train_support_samples, train_support_labels, train_query_samples, train_query_labels


def MAML_train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    global device
    device = torch.device('cpu')
    if config.cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)

    # load data and construct dataset by learn2learn
    FPD_train_tasks, FPD_valid_tasks, FPD_test_tasks = get_FPD_tasks(config)

    print('len(FPD_train_tasks)', len(FPD_train_tasks), 'type(FPD_train_tasks)', type(FPD_train_tasks))
    print('len(FPD_valid_tasks)', len(FPD_valid_tasks), 'type(FPD_valid_tasks)', type(FPD_valid_tasks))
    print('len(FPD_test_tasks)', len(FPD_test_tasks), 'type(FPD_test_tasks)', type(FPD_test_tasks))

    # Create model
    model = BERT.BERT(config)
    model.to(device)
    maml = l2l.algorithms.MAML(model, lr=config.fast_lr, first_order=False)
    optimizer = optim.Adam(maml.parameters(), config.meta_lr)
    loss_CE = nn.CrossEntropyLoss(reduction='mean')

    # 绘图数据
    global step_list, train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list
    step_list = []
    train_loss_list = []
    valid_loss_list = []
    train_accuracy_list = []
    valid_accuracy_list = []

    pbar = tqdm([i for i in range(config.num_iteration)])
    for i in pbar:
        optimizer.zero_grad()

        # 每次循环都取一个batch的tasks来训练，一个batch才更新一次, config.num_iteration决定训练多少个batch
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0

        # 处理一个batch的tasks
        for task_id in range(config.meta_batch_size):
            # Compute meta-train loss
            # model.train()
            learner = maml.clone()
            # model.train()
            learner.train()
            train_task = get_task_data(FPD_train_tasks, config.train_way, config.train_shot, config.train_query)
            train_evaluation_error, train_evaluation_accuracy = fast_adapt(train_task,
                                                                           learner,
                                                                           loss_CE,
                                                                           config.adaptation_step)
            train_evaluation_error.backward()
            meta_train_error += train_evaluation_error.item()
            meta_train_accuracy += train_evaluation_accuracy.item()

            # Compute meta-validation loss
            # model.eval()
            learner = maml.clone()
            # model.eval()
            learner.eval()
            valid_task = get_task_data(FPD_valid_tasks, config.test_way, config.test_shot, config.test_query)
            valid_evaluation_error, valid_evaluation_accuracy = fast_adapt(valid_task,
                                                                           learner,
                                                                           loss_CE,
                                                                           config.adaptation_step)
            meta_valid_error += valid_evaluation_error.item()
            meta_valid_accuracy += valid_evaluation_accuracy.item()

        print()
        print('-----A Batch of Tasks Finish Training-----')
        train_error = meta_train_error / config.meta_batch_size
        train_accuracy = meta_train_accuracy / config.meta_batch_size
        valid_error = meta_valid_error / config.meta_batch_size
        valid_accuracy = meta_valid_accuracy / config.meta_batch_size

        print(
            'Iteration[{}] | Meta Train Accuracy:{:.4f} | Meta Valid Accuracy:{:.4f} | Meta Train Loss:{:.4f} | Meta Valid Loss:{:.4f}'.
                format(i, train_accuracy, valid_accuracy, train_error, valid_error))

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / config.meta_batch_size)
        optimizer.step()

        pbar.set_description('i[{}]'.format(i))
        pbar.set_postfix(accuracy='{0:.4f}'.format(valid_accuracy))

        if i % 1 == 0:
            step_list.append(i)
            train_loss_list.append(train_error)
            valid_loss_list.append(valid_error)
            train_accuracy_list.append(train_accuracy)
            valid_accuracy_list.append(valid_accuracy)

        if i % 50 == 0 and i != 0:
            draw_figure()

    '''
    测试阶段
    '''
    '''Test Period'''
    test_ctr = 0
    test_loss = 0
    test_acc = 0
    for task_id in range(len(FPD_test_tasks)):
        # model.eval()
        learner = maml.clone()
        # model.eval()
        learner.eval()
        test_task = get_task_data(FPD_test_tasks, config.test_way, config.test_shot, config.test_query)
        test_evaluation_error, test_evaluation_accuracy = fast_adapt(test_task,
                                                                     learner,
                                                                     loss_CE,
                                                                     config.adaptation_step)
        test_ctr += 1
        test_loss += test_evaluation_error.item()
        test_acc += test_evaluation_accuracy.item()

        print('task_id[{}] | test_loss={:.4f} | test_acc={:.4f}'
              .format(task_id, test_loss / test_ctr, test_acc / test_ctr))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='meta learning config')
    # 项目参数配置
    parse.add_argument('-path-data', type=str, default='../../../data/task_data/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=False, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0.90, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=0)
    parse.add_argument('-seed', type=int, default=42)
    parse.add_argument('-num_workers', type=int, default=32)

    # 元学习参数配置
    parse.add_argument('-train_way', type=int, default=5)
    parse.add_argument('-train_shot', type=int, default=5)
    parse.add_argument('-train_query', type=int, default=15)
    parse.add_argument('-test_way', type=int, default=5)
    parse.add_argument('-test_shot', type=int, default=5)
    parse.add_argument('-test_query', type=int, default=15)
    parse.add_argument('-meta_batch_size', type=int, default=32)
    parse.add_argument('-adaptation_step', type=int, default=1)
    parse.add_argument('-num_iteration', type=int, default=151)
    # parse.add_argument('-meta_lr', type=float, default=0.001)
    parse.add_argument('-meta_lr', type=float, default=0.01)
    parse.add_argument('-fast_lr', type=float, default=0.4)
    # parse.add_argument('-fast_lr', type=float, default=0.7)
    parse.add_argument('-reg', type=float, default=0.00025, help='weight lambda of regularization')

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
    MAML_train(config)
