# ---encoding:utf-8---
# @Time : 2021.02.25
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : main_MAML.py

import random
import torch
import torchvision
import torch.nn as nn
import learn2learn as l2l
import numpy as np

from configuration import config
from model import CNN
from collections import namedtuple
from util import util_log


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(task, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = task
    data, labels = data.to(device), labels.to(device)

    # print('data.size()', data.size())
    # print('labels.size()', labels.size())

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)

    # indices = np.arange(shots * ways) * 2
    # indices = np.array([x for x in indices if x < len(data)])
    # adaptation_indices[indices] = True

    adaptation_indices[np.arange(shots * ways) * 2] = True

    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)

    print('adaptation_labels', adaptation_labels)
    print('evaluation_labels', evaluation_labels)
    print('predictions', predictions)

    return valid_error, valid_accuracy


if __name__ == '__main__':
    '''
    每一次训练需要主动确定以下变量:
    1.train-name: 本次训练的名称
    2.path-config-data: 模型配置的路径，空字符串''表示加载默认配置
    3.path-train-data: 训练集的路径
    4.path-test-data: 测试集的路径
    
    一次训练对应得到一个result文件夹，以train-name命名，包含:
    1.report: 训练报告
    2.figure: 训练图像
    3.config: 模型配置
    4.model_save: 模型参数
    5.others: 其他数据
    '''

    '''
    设置配置中的必需变量
    '''
    config = config.get_train_config()

    '''
    修改默认配置
    '''
    # num_iterations = config.epoch = 50
    device = torch.device('cuda')

    '''
    设置其他变量
    '''
    num_tasks = 32 * 1000
    train_ways = 5
    train_shots = 5
    test_ways = 5
    test_shots = 15
    meta_lr = 0.0003
    fast_lr = 0.05
    meta_batch_size = 32
    adaptation_steps = 1
    num_iterations = 60000
    cuda = True
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    log = util_log.LOG()

    '''
    加载数据
    '''
    # train = torchvision.datasets.MNIST(root='../data/task_data/MINIST', transform=torchvision.transforms.ToTensor(),
    #                                    train=True, download=True)
    # print('len(train)', len(train), type(train))
    # valid = torchvision.datasets.MNIST(root='../data/task_data/MINIST', transform=torchvision.transforms.ToTensor(),
    #                                    train=True, download=True)
    # print('len(valid)', len(valid), type(valid))
    # test = torchvision.datasets.MNIST(root='../data/task_data/MINIST', transform=torchvision.transforms.ToTensor(),
    #                                   train=True, download=True)
    # print('len(test)', len(test), type(test))

    train = torchvision.datasets.MNIST(root='../data/task_data/MINIST', transform=torchvision.transforms.ToTensor(),
                                       train=True, download=True)
    print('len(train)', len(train), type(train))
    valid = torchvision.datasets.MNIST(root='../data/task_data/MINIST', transform=torchvision.transforms.ToTensor(),
                                       train=True, download=True)
    print('len(valid)', len(valid), type(valid))
    test = torchvision.datasets.MNIST(root='../data/task_data/MINIST', transform=torchvision.transforms.ToTensor(),
                                      train=True, download=True)
    print('len(test)', len(test), type(test))

    '''
    定义任务
    '''
    # train_dataset = l2l.data.MetaDataset(train)
    # valid_dataset = l2l.data.MetaDataset(valid)
    # test_dataset = l2l.data.MetaDataset(test)

    train_dataset = l2l.data.FilteredMetaDataset(train, [0, 1, 2, 8, 9])
    valid_dataset = l2l.data.FilteredMetaDataset(valid, [0, 1, 2, 8, 9])
    test_dataset = l2l.data.FilteredMetaDataset(test, [0, 1, 2, 8, 9])

    # print('len(train_dataset)', len(train_dataset), type(train_dataset))
    print('len(valid_dataset)', len(valid_dataset), type(valid_dataset))
    print('len(test_dataset)', len(test_dataset), type(test_dataset))

    train_tasks = l2l.data.TaskDataset(
        dataset=train_dataset,
        task_transforms=[
            l2l.data.transforms.NWays(train_dataset, n=train_ways),
            l2l.data.transforms.KShots(train_dataset, k=train_shots * 2),
            l2l.data.transforms.LoadData(train_dataset),
        ],
        num_tasks=num_tasks,
    )
    valid_tasks = l2l.data.TaskDataset(
        dataset=valid_dataset,
        task_transforms=[
            l2l.data.transforms.NWays(valid_dataset, n=test_ways),
            l2l.data.transforms.KShots(valid_dataset, k=test_shots * 2),
            l2l.data.transforms.LoadData(valid_dataset),
        ],
        num_tasks=num_tasks,
    )
    test_tasks = l2l.data.TaskDataset(
        dataset=test_dataset,
        task_transforms=[
            l2l.data.transforms.NWays(test_dataset, n=test_ways),
            l2l.data.transforms.KShots(test_dataset, k=test_shots * 2),
            l2l.data.transforms.LoadData(test_dataset),
        ],
        num_tasks=num_tasks,
    )

    print('train_tasks', train_tasks)
    print('train_tasks.sample()', train_tasks.sample()[0].size(), train_tasks.sample()[1].size())
    print('valid_tasks', valid_tasks)
    print('valid_tasks.sample()', valid_tasks.sample()[0].size(), valid_tasks.sample()[1].size())
    print('test_tasks', test_tasks)
    print('test_tasks.sample()', test_tasks.sample()[0].size(), test_tasks.sample()[1].size())

    # 每一个task的data的size是 [ways*shots, channel, height, width]
    # 每一个task的label的size是 [ways*shots]
    BenchmarkTasksets = namedtuple('BenchmarkTasksets', ('train', 'validation', 'test'))
    tasksets = BenchmarkTasksets(train_tasks, valid_tasks, test_tasks)

    '''
    定义模型以及优化器
    '''
    model = CNN.CNN()
    if config.cuda:
        torch.cuda.set_device(config.device)
        model.cuda()

    maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)
    opt = torch.optim.Adam(maml.parameters(), lr=meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    '''
    训练阶段
    '''
    for iteration in range(num_iterations):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()  # each model for each task
            a_task_train = tasksets.train.sample()

            print('a_task_train:', a_task_train[0].size(), a_task_train[1].size(), a_task_train[1])
            # [num_samples = n_way * k_shot * 2, channel, width, height]
            evaluation_error, evaluation_accuracy = fast_adapt(a_task_train,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               train_shots,
                                                               train_ways,
                                                               device)
            # evaluation_error 是内层学习器更新一次参数后的测试损失
            # evaluation_accuracy 是内层学习器更新一次参数后的测试准确率

            evaluation_error.backward()  # 计算的是外层的梯度，会累加一个meta_batch_size的梯度。不能在这里更新
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            a_task_train_valid = tasksets.validation.sample()
            print('a_task_train_valid:', a_task_train_valid[0].size(), a_task_train_valid[1].size(),
                  a_task_train_valid[1])
            # [num_samples = n_way * k_shot * 2, channel, width, height]
            evaluation_error, evaluation_accuracy = fast_adapt(a_task_train_valid,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               test_shots,
                                                               test_ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        # Print some metrics
        train_error = meta_train_error / meta_batch_size
        train_accuracy = meta_train_accuracy / meta_batch_size
        valid_error = meta_valid_error / meta_batch_size
        valid_accuracy = meta_valid_accuracy / meta_batch_size

        print('Iteration[{}]\t| Meta Train Accuracy:{}\t| Meta Valid Accuracy:{}'.
              format(iteration, train_accuracy, valid_accuracy))

        # out_string = '\n' + 'Iteration: {}'.format(iteration) + '\n' + \
        #              'Meta Train Error: {}'.format(train_error) + '\t' + \
        #              'Meta Train Accuracy: {}'.format(train_accuracy) + '\n' + \
        #              'Meta Valid Error: {}'.format(valid_error) + '\t' + \
        #              'Meta Valid Accuracy: {}'.format(valid_accuracy) + '\n'
        # log.Info(out_string)

        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)  # 将一个meta_batch_size的累加梯度取平均
        opt.step()

    '''
    测试阶段
    '''
    iter_num = 1200  # num_sample/(test_shots*test_ways*2)=12000/10=1200
    avg_meta_test_error = 0.0
    avg_meta_test_accuracy = 0.0
    for iteration in range(iter_num):
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = tasksets.test.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               test_shots,
                                                               test_ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()
        test_error = meta_test_error / meta_batch_size
        test_accuracy = meta_test_accuracy / meta_batch_size
        print('Iteration[{}]\t| Meta Test Accuracy:{}\t| Meta Test Accuracy:{}'.
              format(iteration, test_error, test_accuracy))
        avg_meta_test_error += test_error
        avg_meta_test_accuracy = test_accuracy

    print('Finally Average\t| Meta Test Error:{}\t| Meta Test Accuracy:{}'.
          format(avg_meta_test_error / iter_num, avg_meta_test_accuracy / iter_num))
