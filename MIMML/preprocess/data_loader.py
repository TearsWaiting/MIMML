# ---encoding:utf-8---
# @Time : 2020.08.15
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : data_loader.py


import pickle
import torch
import torch.utils.data as Data

from configuration import config
from util import util_file
from preprocess import data_augmentation


def transform_token2index(sequences, config):
    token2index = config.token2index

    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)

    print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
    print('sequences_residue', sequences[0:5])
    print('token_list', token_list[0:5])
    return token_list, max_len


def make_data_with_unified_length(token_list, labels, config):
    padded_max_len = config.max_len
    token2index = config.token2index

    data = []
    for i in range(len(labels)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]
        n_pad = padded_max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append([token_list[i], labels[i]])

    print('-' * 20, '[make_data_with_unified_length]: check token_list head', '-' * 20)
    print('padded_max_len', padded_max_len)
    print('token_list + [pad]', token_list[0:5])
    return data


# 构造迭代器
def construct_dataset(data, config):
    cuda = config.cuda
    batch_size = config.batch_size

    # print('-' * 20, '[construct_dataset]: check data dimension', '-' * 20)
    # print('len(data)', len(data))
    # print('len(data[0])', len(data[0]))
    # print('len(data[0][0])', len(data[0][0]))
    # print('data[0][1]', data[0][1])
    # print('len(data[1][0])', len(data[1][0]))
    # print('data[1][1]', data[1][1])

    input_ids, labels = zip(*data)

    if cuda:
        input_ids, labels = torch.cuda.LongTensor(input_ids), torch.cuda.LongTensor(labels)
    else:
        input_ids, labels = torch.LongTensor(input_ids), torch.LongTensor(labels)

    print('-' * 20, '[construct_dataset]: check data device', '-' * 20)
    print('input_ids.device:', input_ids.device)
    print('labels.device:', labels.device)

    print('-' * 20, '[construct_dataset]: check data shape', '-' * 20)
    print('input_ids:', input_ids.shape)  # [num_sequences, seq_len]
    print('labels:', labels.shape)  # [num_sequences, seq_len]

    data_loader = Data.DataLoader(MyDataSet(input_ids, labels),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)

    print('len(data_loader)', len(data_loader))
    return data_loader


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def load_data(config):
    path_data_train = config.path_train_data
    path_data_test = config.path_test_data

    # data augmentation
    # sequences_train, labels_train = data_augmentation.augmentation(path_data_train, config, append = False)

    sequences_train, labels_train = util_file.load_tsv_format_data(path_data_train)
    if path_data_test is not None:
        sequences_test, labels_test = util_file.load_tsv_format_data(path_data_test)
    # sequences_train: ['MNH', 'APD', ...]
    # labels_train: [1, 0, ...]

    token_list_train, max_len_train = transform_token2index(sequences_train, config)
    if path_data_test is not None:
        token_list_test, max_len_test = transform_token2index(sequences_test, config)
    else:
        max_len_test = 0
    # token_list_train: [[1, 5, 8], [2, 7, 9], ...]

    config.max_len = max(max_len_train, max_len_test)
    config.max_len_train = max_len_train
    config.max_len_test = max_len_test
    config.max_len = config.max_len + 2  # add [CLS] and [SEP]
    print('max_len_train', max_len_train)
    print('max_len_test', max_len_test)

    data_train = make_data_with_unified_length(token_list_train, labels_train, config)
    if path_data_test is not None:
        data_test = make_data_with_unified_length(token_list_test, labels_test, config)
    # data_train: [[[1, 5, 8], 0], [[2, 7, 9], 1], ...]

    data_loader_train = construct_dataset(data_train, config)
    if path_data_test is not None:
        data_loader_test = construct_dataset(data_test, config)
    else:
        data_loader_test = None

    return data_loader_train, data_loader_test


def get_finetune_dataset(config):
    path_data_train = config.path_train_data
    path_data_test = config.path_test_data

    sequences, labels = util_file.load_tsv_format_data(path_data_train)

    if path_data_test is not None:
        sequences_test, labels_test = util_file.load_tsv_format_data(path_data_test)
        # sequences_train: ['MNH', 'APD', ...]
        # labels_train: [1, 0, ...]

        sequences = sequences + sequences_test
        labels = labels + labels_test

    token_list, max_len = transform_token2index(sequences, config)
    # token_list: [[1, 5, 8], [2, 7, 9], ...]

    print('max_len', config.max_len)

    data = make_data_with_unified_length(token_list, labels, config)
    # data: [[[1, 5, 8], 0], [[2, 7, 9], 1], ...]

    # construct dataset
    cuda = config.cuda
    input_ids, labels = zip(*data)

    if cuda:
        input_ids, labels = torch.cuda.LongTensor(input_ids), torch.cuda.LongTensor(labels)
    else:
        input_ids, labels = torch.LongTensor(input_ids), torch.LongTensor(labels)

    print('-' * 20, '[construct_dataset]: check data device', '-' * 20)
    print('input_ids.device:', input_ids.device)
    print('labels.device:', labels.device)

    print('-' * 20, '[construct_dataset]: check data shape', '-' * 20)
    print('input_ids:', input_ids.shape)  # [num_sequences, seq_len]
    print('labels:', labels.shape)  # [num_sequences, seq_len]

    dataset = MyDataSet(input_ids, labels)
    return dataset


if __name__ == '__main__':
    '''
    check loading tsv data
    '''
    config = config.get_train_config()

    token2index = pickle.load(open('../data/residue2idx.pkl', 'rb'))
    config.token2index = token2index
    print('token2index', token2index)

    config.path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-80-train.tsv'
    sequences, labels = util_file.load_tsv_format_data(config.path_train_data)
    token_list, max_len = transform_token2index(sequences, config)
    data = make_data_with_unified_length(token_list, labels, config)
    data_loader = construct_dataset(data, config)

    print('-' * 20, '[data_loader]: check data batch', '-' * 20)
    for i, batch in enumerate(data_loader):
        input, label = batch
        print('batch[{}], input:{}, label:{}'.format(i, input.shape, label.shape))
