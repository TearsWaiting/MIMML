# ---encoding:utf-8---
# @Time : 2021.03.26
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : protonet_miniImagenet.py

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import learn2learn as l2l

from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import miniimagenet
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


def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return logits


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.ConvBase(output_size=z_dim,
                                                  hidden=hid_dim,
                                                  channels=x_dim,
                                                  max_pool=True)
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


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


def fast_adapt(model, task, way, shot):
    support_samples, support_labels, query_samples, query_labels = task
    # print('support_labels', support_labels)

    support_embeddings = model(support_samples)
    query_embeddings = model(query_samples)
    query_labels = query_labels.long()

    # sort 否则不收敛
    support_sort = torch.sort(support_labels)
    support_embeddings = torch.index_select(support_embeddings, 0, support_sort.indices)
    support_labels = torch.index_select(support_labels, 0, support_sort.indices)
    support_embeddings = support_embeddings.reshape(way, shot, -1).mean(dim=1)

    query_sort = torch.sort(query_labels)
    query_embeddings = torch.index_select(query_embeddings, 0, query_sort.indices)
    query_labels = torch.index_select(query_labels, 0, query_sort.indices)

    logits = pairwise_distances_logits(query_embeddings, support_embeddings)
    loss = F.cross_entropy(logits, query_labels)
    acc = accuracy(logits, query_labels)
    return loss, acc


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='meta learning config')

    # 项目配置参数
    parse.add_argument('-path-data', type=str, default='../../data/task_data/', help='保存字典的位置')
    parse.add_argument('-save-best', type=bool, default=False, help='当得到更好的准确度是否要保存')
    parse.add_argument('-threshold', type=float, default=0.90, help='准确率阈值')
    parse.add_argument('-cuda', type=bool, default=True)
    parse.add_argument('-device', type=int, default=1)
    parse.add_argument('-seed', type=int, default=43)
    parse.add_argument('-num_workers', type=int, default=4)

    parse.add_argument('-max-epoch', type=int, default=1)
    parse.add_argument('-meta-batch-size', type=int, default=100)
    parse.add_argument('-train-way', type=int, default=20)
    parse.add_argument('-train-shot', type=int, default=5)
    parse.add_argument('-train-query', type=int, default=15)
    parse.add_argument('-test-way', type=int, default=5)
    parse.add_argument('-test-shot', type=int, default=5)
    parse.add_argument('-test-query', type=int, default=15)

    config = parse.parse_args()
    print(config)

    device = torch.device('cpu')
    if config.cuda and torch.cuda.device_count():
        print("Using gpu")
        torch.cuda.manual_seed(config.seed)
        device = torch.device('cuda')
        torch.cuda.set_device(config.device)

    model = Convnet()
    model.to(device)

    # load data and construct dataset by learn2learn
    dataset_train = miniimagenet(config.path_data,
                                 shots=config.train_shot,
                                 ways=config.train_way,
                                 shuffle=True,
                                 test_shots=config.train_query,
                                 meta_train=True,
                                 download=False)
    dataset_validation = miniimagenet(config.path_data,
                                      shots=config.test_shot,
                                      ways=config.test_way,
                                      shuffle=True,
                                      test_shots=config.test_query,
                                      meta_val=True,
                                      download=False)
    dataset_test = miniimagenet(config.path_data,
                                shots=config.test_shot,
                                ways=config.test_way,
                                shuffle=True,
                                test_shots=config.test_query,
                                meta_test=True,
                                download=False)

    dataloader_train = BatchMetaDataLoader(dataset_train,
                                           batch_size=config.meta_batch_size,
                                           shuffle=True,
                                           num_workers=config.num_workers)
    dataloader_validation = BatchMetaDataLoader(dataset_validation,
                                                batch_size=30,
                                                shuffle=True,
                                                num_workers=config.num_workers)
    dataloader_test = BatchMetaDataLoader(dataset_test,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=config.num_workers)

    print('len(dataloader_train)', len(dataloader_train), 'type(dataloader_train)', type(dataloader_train))
    print('len(dataloader_validation)', len(dataloader_validation), 'type(dataloader_validation)',
          type(dataloader_validation))
    print('len(dataloader_test)', len(dataloader_test), 'type(dataloader_test)', type(dataloader_test))

    generator_dataloader_validation = batch_task_generator(dataloader_validation)
    generator_dataloader_test = batch_task_generator(dataloader_test)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    # 绘图数据
    # global step_list, train_loss_list, valid_loss_list, train_accuracy_list, valid_accuracy_list
    step_list = []
    train_loss_list = []
    valid_loss_list = []
    train_accuracy_list = []
    valid_accuracy_list = []

    '''开始训练'''
    with tqdm(dataloader_train, total=config.max_epoch) as pbar:
        # 每次循环都取一个batch的tasks来训练，一个batch才更新一次, config.max_epoch决定训练多少个batch
        # 每一个batch任务里面的batch['train']和batch['test']是标签对齐的，属于同一个任务，只是样本不同
        # batch['train']是support set, batch['test']是query set
        print()
        print('=' * 100)

        # 遍历元训练集
        for epoch, task_batch_train in enumerate(pbar):
            # 获取元训练任务
            train_support_inputs, train_support_targets, train_query_inputs, train_query_targets = \
                get_batch_task_data(task_batch_train)

            # 获取元验证任务
            try:
                task_batch_valid = next(generator_dataloader_validation)
            except StopIteration:
                generator_dataloader_validation = batch_task_generator(dataloader_validation)
                task_batch_valid = next(generator_dataloader_validation)

            if len(task_batch_valid['train'][1]) < 30:
                # print('len(task_batch_valid[\'train\'][1])', len(task_batch_valid['train'][1]))
                generator_dataloader_validation = batch_task_generator(dataloader_validation)
                task_batch_valid = next(generator_dataloader_validation)
            valid_support_inputs, valid_support_targets, valid_query_inputs, valid_query_targets = \
                get_batch_task_data(task_batch_valid)

            '''Train Period'''
            model.train()
            train_ctr = 0
            train_loss = 0
            train_acc = 0
            for task_id in range(config.meta_batch_size):
                train_support_samples = train_support_inputs[task_id]
                train_support_labels = train_support_targets[task_id]
                train_query_samples = train_query_inputs[task_id]
                train_query_labels = train_query_targets[task_id]

                train_task = [train_support_samples, train_support_labels, train_query_samples, train_query_labels]
                loss, acc = fast_adapt(model, train_task, config.train_way, config.train_shot)

                train_ctr += 1
                train_loss += loss.item()
                train_acc += acc.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            '''Validation Period'''
            model.eval()
            valid_ctr = 0
            valid_loss = 0
            valid_acc = 0
            for task_id in range(30):
                valid_support_samples = valid_support_inputs[task_id]
                valid_support_labels = valid_support_targets[task_id]
                valid_query_samples = valid_query_inputs[task_id]
                valid_query_labels = valid_query_targets[task_id]
                valid_task = [valid_support_samples, valid_support_labels, valid_query_samples, valid_query_labels]
                loss, acc = fast_adapt(model, valid_task, config.test_way, config.test_shot)

                valid_ctr += 1
                valid_loss += loss.item()
                valid_acc += acc.item()

            print()
            print('epoch[{}] | train_loss={:.4f} | train_acc={:.4f} | valid_loss={:.4f} | valid_acc={:.4f}'.format(
                epoch, train_loss / train_ctr, train_acc / train_ctr, valid_loss / valid_ctr, valid_acc / valid_ctr))

            pbar.set_description('epoch[{}]'.format(epoch))
            pbar.set_postfix(accuracy='{0:.4f}'.format(valid_acc / valid_ctr))

            if epoch % 1 == 0:
                step_list.append(epoch)
                train_loss_list.append(train_loss / train_ctr)
                valid_loss_list.append(valid_loss / valid_ctr)
                train_accuracy_list.append(train_acc / train_ctr)
                valid_accuracy_list.append(valid_acc / valid_ctr)

            if epoch % 10 == 0 and epoch != 0:
                draw_figure()

            if epoch + 1 == config.max_epoch:
                break

        print('=' * 100)
        print('Train Over')

    '''Test Period'''
    test_ctr = 0
    test_acc = 0

    for task_id, task_batch_test in enumerate(dataloader_test, 1):
        test_support_inputs, test_support_targets, test_query_inputs, test_query_targets = \
            get_batch_task_data(task_batch_test)
        test_support_samples = test_support_inputs[0]
        test_support_labels = test_support_targets[0]
        test_query_samples = test_query_inputs[0]
        test_query_labels = test_query_targets[0]
        test_task = [test_support_samples, test_support_labels, test_query_samples, test_query_labels]
        loss, acc = fast_adapt(model, test_task, config.test_way, config.test_shot)
        test_ctr += 1
        test_acc += acc.item()
        print('task_id[{}] | ACC = {:.2f}% ({:.2f}%)'.format(task_id, test_acc / test_ctr * 100, acc * 100))
