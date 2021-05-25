# ---encoding:utf-8---
# @Time : 2021.03.07
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : maml_torchmeta_learn2learn.py


import random
import numpy as np
import argparse
import torch
import learn2learn as l2l
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from torch import nn, optim
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import miniimagenet


def get_accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(task, learner, loss, adaptation_steps, shots, ways, device):
    support_samples, support_labels, query_samples, query_labels = task
    support_samples, support_labels = support_samples.to(device), support_labels.to(device)
    query_samples, query_labels = query_samples.to(device), query_labels.to(device)

    # print('support_labels', support_labels)
    # print('support_samples', support_samples)
    # print('learner(support_samples)', learner(support_samples))

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(support_samples), support_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(query_samples)
    evaluation_error = loss(predictions, query_labels)
    evaluation_accuracy = get_accuracy(predictions, query_labels)
    return evaluation_error, evaluation_accuracy


def batch_task_generator(dataloader):
    for i, data in enumerate(dataloader):
        yield data


def get_batch_task_data(batch):
    support_inputs, support_targets = batch['train']
    support_inputs = support_inputs.to(device=device)
    support_targets = support_targets.to(device=device)
    # support_inputs: [task_batch_size, support_num_samples = n_ways * k_support_shots, channel, width, height]
    # support_targets: [task_batch_size, support_num_samples = n_ways * k_support_shots]

    query_inputs, query_targets = batch['test']
    query_inputs = query_inputs.to(device=device)
    query_targets = query_targets.to(device=device)
    # query_inputs: [task_batch_size, query_num_samples = n_ways * k_query_shots, channel, width, height]
    # query_targets: [task_batch_size, query_num_samples = n_ways * k_query_shots]

    return support_inputs, support_targets, query_inputs, query_targets


def draw_figure():
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    plt.subplot(2, 2, 1)
    plt.title("Average Inner Loss", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_list, train_loss_list)
    plt.subplot(2, 2, 2)
    plt.title("Average Outer Loss", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_list, valid_loss_list)
    plt.subplot(2, 2, 3)
    plt.title("Inner Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_list, train_accuracy_list)
    plt.subplot(2, 2, 4)
    plt.title("Outer Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_list, valid_accuracy_list)

    # plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


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
    dataset_train = miniimagenet(config.path_data,
                                 shots=config.shots,
                                 ways=config.ways,
                                 shuffle=True,
                                 test_shots=config.test_shots,
                                 meta_train=True,
                                 download=False)
    dataset_validation = miniimagenet(config.path_data,
                                      shots=config.shots,
                                      ways=config.ways,
                                      shuffle=True,
                                      test_shots=config.test_shots,
                                      meta_val=True,
                                      download=False)
    dataset_test = miniimagenet(config.path_data,
                                shots=config.shots,
                                ways=config.ways,
                                shuffle=True,
                                test_shots=config.test_shots,
                                meta_test=True,
                                download=False)

    dataloader_train = BatchMetaDataLoader(dataset_train,
                                           batch_size=config.meta_batch_size,
                                           shuffle=True,
                                           num_workers=config.num_workers)
    dataloader_validation = BatchMetaDataLoader(dataset_validation,
                                                batch_size=config.meta_batch_size,
                                                shuffle=True,
                                                num_workers=config.num_workers)
    dataloader_test = BatchMetaDataLoader(dataset_test,
                                          batch_size=config.meta_batch_size,
                                          shuffle=True,
                                          num_workers=config.num_workers)

    print('len(dataloader_train)', len(dataloader_train), 'type(dataloader_train)', type(dataloader_train))
    print('len(dataloader_validation)', len(dataloader_validation), 'type(dataloader_validation)',
          type(dataloader_validation))
    print('len(dataloader_test)', len(dataloader_test), 'type(dataloader_test)', type(dataloader_test))

    generator_dataloader_validation = batch_task_generator(dataloader_validation)
    generator_dataloader_test = batch_task_generator(dataloader_test)

    # Create model
    model = l2l.vision.models.MiniImagenetCNN(config.ways)
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

    with tqdm(dataloader_train, total=config.num_iteration) as pbar:
        # 每次循环都取一个batch的tasks来训练，一个batch才更新一次, config.num_iteration决定训练多少个batch
        # 每一个batch任务里面的batch['train']和batch['test']是标签对齐的，属于同一个任务，只是样本不同
        # batch['train']是support set, batch['test']是query set
        print('=' * 100)

        # 遍历元训练集
        for task_batch_idx, task_batch_train in enumerate(pbar):
            model.zero_grad()

            # 获取元训练任务
            train_support_inputs, train_support_targets, train_query_inputs, train_query_targets = \
                get_batch_task_data(task_batch_train)
            # 获取元验证任务
            try:
                task_batch_valid = next(generator_dataloader_validation)
            except StopIteration:
                generator_dataloader_validation = batch_task_generator(dataloader_validation)
                task_batch_valid = next(generator_dataloader_validation)

            if len(task_batch_valid['train'][1]) < config.meta_batch_size:
                print('len(task_batch_valid[\'train\'][1])', len(task_batch_valid['train'][1]))
                generator_dataloader_validation = batch_task_generator(dataloader_validation)
                task_batch_valid = next(generator_dataloader_validation)

            valid_support_inputs, valid_support_targets, valid_query_inputs, valid_query_targets = \
                get_batch_task_data(task_batch_valid)

            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            # 处理一个batch的tasks
            for task_id in range(config.meta_batch_size):
                # Compute meta-train loss
                train_support_samples = train_support_inputs[task_id]
                train_support_labels = train_support_targets[task_id]
                train_query_samples = train_query_inputs[task_id]
                train_query_labels = train_query_targets[task_id]

                # 这里特殊，不需要自行划分support set和query set，也不需要做标签映射

                train_task = [train_support_samples, train_support_labels, train_query_samples, train_query_labels]
                learner = maml.clone()
                train_evaluation_error, train_evaluation_accuracy = fast_adapt(train_task,
                                                                               learner,
                                                                               loss_CE,
                                                                               config.adaptation_step,
                                                                               config.shots,
                                                                               config.ways,
                                                                               device)
                train_evaluation_error.backward()
                meta_train_error += train_evaluation_error.item()
                meta_train_accuracy += train_evaluation_accuracy.item()

                # Compute meta-validation loss
                valid_support_samples = valid_support_inputs[task_id]
                valid_support_labels = valid_support_targets[task_id]
                valid_query_samples = valid_query_inputs[task_id]
                valid_query_labels = valid_query_targets[task_id]

                # 这里特殊，不需要自行划分support set和query set，也不需要做标签映射

                valid_task = [valid_support_samples, valid_support_labels, valid_query_samples, valid_query_labels]
                learner = maml.clone()
                valid_evaluation_error, valid_evaluation_accuracy = fast_adapt(valid_task,
                                                                               learner,
                                                                               loss_CE,
                                                                               config.adaptation_step,
                                                                               config.shots,
                                                                               config.ways,
                                                                               device)
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
                    format(task_batch_idx, train_accuracy, valid_accuracy, train_error, valid_error))

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / config.meta_batch_size)
            optimizer.step()

            pbar.set_description('task_batch_idx[{}]'.format(task_batch_idx))
            pbar.set_postfix(accuracy='{0:.4f}'.format(valid_accuracy))

            if task_batch_idx % 1 == 0:
                step_list.append(task_batch_idx)
                train_loss_list.append(train_error)
                valid_loss_list.append(valid_error)
                train_accuracy_list.append(train_accuracy)
                valid_accuracy_list.append(valid_accuracy)

            if task_batch_idx % 100 == 0 and task_batch_idx != 0:
                draw_figure()

            if task_batch_idx + 1 ==  config.num_iteration:
                break

    '''
    测试阶段
    '''
    iter_num = 1200
    avg_meta_test_error = 0.0
    avg_meta_test_accuracy = 0.0
    for iteration in range(iter_num):
        # 获取元测试任务
        try:
            task_batch_test = next(generator_dataloader_test)
        except StopIteration:
            generator_dataloader_test = batch_task_generator(generator_dataloader_test)
            task_batch_test = next(generator_dataloader_test)

        test_support_inputs, test_support_targets, test_query_inputs, test_query_targets = \
            get_batch_task_data(task_batch_test)

        meta_test_error = 0.0
        meta_test_accuracy = 0.0

        for task in range(len(task_batch_test)):
            # Compute meta-testing loss
            test_support_samples = test_support_inputs[task_id]
            test_support_labels = test_support_targets[task_id]
            test_query_samples = test_query_inputs[task_id]
            test_query_labels = test_query_targets[task_id]

            # 这里特殊，不需要自行划分support set和query set，也不需要做标签映射

            test_task = [test_support_samples, test_support_labels, test_query_samples, test_query_labels]
            learner = maml.clone()
            test_evaluation_error, test_evaluation_accuracy = fast_adapt(test_task,
                                                                         learner,
                                                                         loss_CE,
                                                                         config.adaptation_step,
                                                                         config.shots,
                                                                         config.ways,
                                                                         device)
            meta_test_error += test_evaluation_error.item()
            meta_test_accuracy += test_evaluation_accuracy.item()

        test_error = meta_test_error / config.meta_batch_size
        test_accuracy = meta_test_accuracy / config.meta_batch_size

        print('Iteration[{}]\t| Meta Test Accuracy:{:.4f}\t| Meta Test Accuracy:{:.4f}'.
              format(iteration, test_error, test_accuracy))

        avg_meta_test_error += test_error
        avg_meta_test_accuracy = test_accuracy

    print('Final Average\t| Meta Test Error:{:.4f}\t| Meta Test Accuracy:{:.4f}'.
          format(avg_meta_test_error / iter_num, avg_meta_test_accuracy / iter_num))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='meta learning config')
    # 项目配置参数
    parse.add_argument('-path-data', type=str, default='../../../data/task_data/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=False, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0.90, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=0)
    parse.add_argument('-seed', type=int, default=42)
    parse.add_argument('-num_workers', type=int, default=16)

    # 元学习配置参数
    parse.add_argument('-ways', type=int, default=5)
    parse.add_argument('-shots', type=int, default=5)
    parse.add_argument('-test_shots', type=int, default=15)
    parse.add_argument('-meta_batch_size', type=int, default=32)
    parse.add_argument('-adaptation_step', type=int, default=1)
    parse.add_argument('-num_iteration', type=int, default=20000)
    parse.add_argument('-meta_lr', type=float, default=0.003)
    parse.add_argument('-fast_lr', type=float, default=0.5)

    config = parse.parse_args()
    MAML_train(config)
