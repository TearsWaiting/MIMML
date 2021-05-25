import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from torchmeta.datasets.helpers import omniglot, miniimagenet, tieredimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy

logger = logging.getLogger(__name__)


def draw_figure():
    sns.set(style="darkgrid")
    plt.figure(22, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    plt.subplot(2, 2, 1)
    plt.title("Average Inner Loss", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_list, inner_loss_list)
    plt.subplot(2, 2, 2)
    plt.title("Average Outer Loss", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.plot(step_list, outer_loss_list)
    plt.subplot(2, 2, 3)
    plt.title("Inner Acc Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_list, train_accuracy_list)
    plt.subplot(2, 2, 4)
    plt.title("Outer Loss Curve", fontsize=23)
    plt.xlabel("Step", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.plot(step_list, test_accuracy_list)

    # plt.savefig(config.result_folder + '/' + fig_name + '.png')
    plt.show()


def train(args):
    logger.warning('This script is an github_project to showcase the MetaModule and '
                   'data-loading features of Torchmeta, and as such has been '
                   'very lightly tested. For a better tested implementation of '
                   'Model-Agnostic Meta-Learning (MAML) using Torchmeta with '
                   'more features (including multi-step adaptation and '
                   'different datasets), please check `https://github.com/'
                   'tristandeleu/pytorch-maml`.')

    # dataset = omniglot(args.folder,
    #                    shots=args.num_shots,
    #                    ways=args.num_ways,
    #                    shuffle=True,
    #                    test_shots=15,
    #                    meta_train=True,
    #                    download=args.download)
    dataset = miniimagenet(args.folder,
                           shots=args.num_shots,
                           ways=args.num_ways,
                           shuffle=True,
                           test_shots=15,
                           meta_train=True,
                           download=args.download)
    # dataset = tieredimagenet(args.folder,
    #                          shots=args.num_shots,
    #                          ways=args.num_ways,
    #                          shuffle=True,
    #                          test_shots=15,
    #                          meta_train=True,
    #                          download=args.download)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    # 如果要获取验证集和测试集
    # miniimagenet: 一共100个classes, 64个base classes, 16个validation classes, 20个novel classes
    # dataset_val = miniimagenet(args.folder,
    #                        shots=args.num_shots,
    #                        ways=args.num_ways,
    #                        shuffle=True,
    #                        test_shots=15,
    #                        meta_val=True,
    #                        download=args.download)
    #
    # dataloader_val = BatchMetaDataLoader(dataset_val,
    #                                  batch_size=args.batch_size,
    #                                  shuffle=True,
    #                                  num_workers=args.num_workers)
    #
    # dataset_test = miniimagenet(args.folder,
    #                        shots=args.num_shots,
    #                        ways=args.num_ways,
    #                        shuffle=True,
    #                        test_shots=5,
    #                        meta_test=True,
    #                        download=args.download)

    # model = ConvolutionalNeuralNetwork(1, args.num_ways, hidden_size=args.hidden_size)
    model = ConvolutionalNeuralNetwork(3, args.num_ways, hidden_size=args.hidden_size)

    model.to(device=args.device)
    model.train()
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    global step_list, inner_loss_list, outer_loss_list, train_accuracy_list, test_accuracy_list
    step_list = []
    inner_loss_list = []
    outer_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []

    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        # 每次循环都取一个batch的tasks来训练，一个batch才更新一次, args.num_batches决定num_iteration
        # 每一个batch任务里面的 batch['train']，batch['test']是标签对齐的，属于同一个任务，只是样本不同
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            # print('train_inputs.size()', train_inputs.size())
            # [task_batch_size, train_num_samples = n_ways * k_train_shots, channel, width, height]
            # print('train_targets.size()', train_targets.size())
            # print('train_targets', train_targets)
            # [task_batch_size, train_num_samples = n_ways * k_train_shots]

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            # print('test_inputs.size()', test_inputs.size())
            # [task_batch_size, test_num_samples = n_ways * k_test_shots, channel, width, height]
            # print('test_targets.size()', test_targets.size())
            # print('test_targets', test_targets)
            # [task_batch_size, test_num_samples = n_way * k_test_shots]

            avg_inner_loss = torch.tensor(0., device=args.device)  # 记录内层学习器/任务的总损失的平均（一个batch的平均内层损失）
            avg_outer_loss = torch.tensor(0., device=args.device)  # 记录外层学习器/任务的总损失的平均（一个batch的平均外层损失）
            train_accuracy = torch.tensor(0., device=args.device)  # 记录内层学习器更新一次参数时的训练准确率
            test_accuracy = torch.tensor(0., device=args.device)  # 记录内层学习器更新一次参数后的测试准确率

            for task_idx, (train_input, train_target, test_input, test_target) in enumerate(
                    zip(train_inputs, train_targets, test_inputs, test_targets)):
                # print('task_idx', task_idx)
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)  # 内层学习器/任务更新一次参数时的的训练损失
                avg_inner_loss += inner_loss

                with torch.no_grad():
                    train_accuracy += get_accuracy(train_logit, train_target)  # 内层学习器/任务更新一次参数时的的训练准确率

                # 内层任务的模型参数更新
                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=args.step_size,
                                                    first_order=args.first_order)

                # 内层任务的测试
                test_logit = model(test_input, params=params)
                outer_loss = F.cross_entropy(test_logit, test_target)  # 内层学习器/任务新一次参数后的测试损失
                avg_outer_loss += outer_loss

                with torch.no_grad():
                    test_accuracy += get_accuracy(test_logit, test_target)  # 内层学习器/任务新一次参数后的测试准确率

            '''Inner Loop Over. Meta Parameters Optimization'''
            avg_inner_loss.div_(args.batch_size)
            avg_outer_loss.div_(args.batch_size)
            train_accuracy.div_(args.batch_size)
            test_accuracy.div_(args.batch_size)

            # 元学习器更新参数
            avg_outer_loss.backward()
            meta_optimizer.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(test_accuracy.item()))
            print()


            if batch_idx % 1 == 0:
                step_list.append(batch_idx)
                inner_loss_list.append(avg_inner_loss.cpu().detach())
                outer_loss_list.append(avg_outer_loss.cpu().detach())
                train_accuracy_list.append(train_accuracy.cpu().detach())
                test_accuracy_list.append(test_accuracy.cpu().detach())

            if batch_idx >= args.num_batches:
                break

            if batch_idx % 100 == 0:
                draw_figure()


    # Save model
    if args.output_folder is not None:
        filename = os.path.join(args.output_folder, 'maml_omniglot_'
                                                    '{0}shot_{1}way.th'.format(args.num_shots, args.num_ways))
        with open(filename, 'wb') as f:
            state_dict = model.state_dict()
            torch.save(state_dict, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Model-Agnostic Meta-Learning (MAML)')

    parser.add_argument('--folder', type=str, default='../../../data/task_data/',
                        help='Path to the folder the data is downloaded to.')

    # parser.add_argument('folder', type=str,
    #     help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--first-order', action='store_true',
                        help='Use the first-order approximation of MAML.')
    parser.add_argument('--step-size', type=float, default=0.4,
                        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=20,
                        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Number of tasks in a mini-batch of tasks (default: 16).')
    parser.add_argument('--num-batches', type=int, default=30000,
                        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true', default=False,
                        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
                                         and torch.cuda.is_available() else 'cpu')

    print('args.download', args.download)
    print('args.use_cuda', args.use_cuda)
    print('torch.cuda.is_available()', torch.cuda.is_available())
    print('args.device', args.device)
    torch.cuda.set_device(0)

    train(args)
