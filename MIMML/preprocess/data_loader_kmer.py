# ---encoding:utf-8---
# @Time : 2020.08.15
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : data_loader_kmer.py


import pickle
import torch
import torch.utils.data as Data

from configuration import config
from util import util_file

'''
该模块用于读取并处理用于模型微调的数据，即加载目标任务的数据（不处理预训练数据）
'''


def split_kmer(sequences, k_mer):
    print('=' * 50, '1 to {}-mer Split'.format(k_mer), '=' * 50)
    '''
    sequences=['MKTLLLTL', 'VVVTIVC']
    1-mer:[
            [
             ['M', 'K', 'T', 'L', 'L', 'L', 'T', 'L']
            ], 
            [
             ['V', 'V', 'V', 'T', 'I', 'V', 'C']
            ]
          ]
    2-mer:[
            [
             ['-M', 'KT', 'LL', 'LT', 'L-'], 
             ['MK', 'TL', 'LL', 'TL']
            ], 
            [
             ['-V', 'VV', 'TI', 'VC'], 
             ['VV', 'VT', 'IV', 'C-']
            ]
          ]
    3-mer:[
            [
             ['--M', 'KTL', 'LLT', 'L--'], 
             ['-MK', 'TLL', 'LTL'],
             ['MKT', 'LLL', 'TL-']
            ], 
            [
             ['--V', 'VVT', 'IVC'], 
             ['-VV', 'VTI', 'VC-'],
             ['VVV', 'TIV', 'V--']
            ]
          ]
    '''
    sequences_kmer = []

    for seq in sequences:
        kmer_list = [[] for i in range(k_mer)]
        seq_kmer = [[] for i in range(len(seq))]

        # Traverse every token of every sequence
        for i in range(len(seq)):
            # Each token is divided into k-mer from 1 to k
            for k in range(1, k_mer + 1):
                # Each token position is represented by k kinds of kmer
                for j in range(k):
                    # There is no need to add '-' before the beginning of the sequence
                    if i - j >= 0:
                        kmer = seq[i - j:i - j + k]

                        # Need to add '-' after the end of the sequence
                        if i - j + k > len(seq):
                            num_pad = i - j + k - len(seq)
                            kmer += '-' * num_pad

                    # Need to add '-' before the beginning of the sequence
                    else:
                        num_pad = j - i
                        kmer = ('-' * num_pad) + seq[0:  k - num_pad]

                        # Need to add '-' after the end of the sequence
                        if k - num_pad > len(seq):
                            num_pad = k - num_pad - len(seq)
                            kmer += '-' * num_pad

                    kmer_list[k - 1].append(kmer)
                    seq_kmer[i].append(kmer)
        sequences_kmer.append(seq_kmer)

        if len(sequences_kmer) % 1000 == 0:
            print('Processing: {}/{}'.format(len(sequences_kmer), len(sequences)))

    print('=' * 50, '{}-mer Split Over'.format(k_mer), '=' * 50)
    return sequences_kmer


def merge_residue_set(sequences):
    print('=' * 50, 'merge_residue_set', '=' * 50)
    # Reduced 4: (FWY, CILMV, AGPST, DEHKNQR)
    merge_list = ['FWY', 'CILMV', 'AGPST', 'DEHKNQR']
    target_token_list = ['F', 'C', 'A', 'D']
    for i, residue_set_str in enumerate(merge_list):
        merge_list[i] = list(residue_set_str)

    for i, seq in enumerate(sequences):
        for j, residue in enumerate(seq):
            token = sequences[i][j]
            for k, residue_set in enumerate(merge_list):
                if token in residue_set:
                    sequences[i][j] = target_token_list[k]
    return sequences


def transform_token2index(sequences, config):
    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)
    print('sequences_residue_sample', sequences[0:5])  # 完整的蛋白质序列->氨基酸残基列表
    # [[MKTLLLTL], [VVVTIVC]] -> [['M', 'K', 'T', 'L', 'L', 'L', 'T', 'L'], ['V', 'V', 'V', 'T', 'I', 'V', 'C']]

    for i, seq in enumerate(sequences):
        sequences[i] = ''.join(seq)

    k_mer = config.k_mer
    token2index = config.token2index
    sequences_kmer = split_kmer(sequences, k_mer)

    '''
    Level 1: for all sequences, the dimensions is num_seq
    Layer 2: corresponding to each token position of each sequence, the dimensions is seq_len
    Layer 3: k k-mer representations corresponding to each token position, 
    each kmer representation contains k dislocation forms, 
    and the dimensions is kmer_num = 1 + 2 +... + k, sum of arithmetic sequence
    '''

    new_token_list = []
    num_token2index = len(token2index)

    token_list = list()
    max_len = 0
    for seq_kmer in sequences_kmer:
        seq_kmer_id_list = []
        for kmer_list in seq_kmer:
            # Handle keys not in token2index
            for kmer in kmer_list:
                if kmer not in token2index:
                    new_token_list.append(kmer)
            for i, token in enumerate(new_token_list):
                token2index[token] = i + num_token2index

            kmer_id_list = [token2index[kmer] for kmer in kmer_list]
            seq_kmer_id_list.append(kmer_id_list)
        token_list.append(seq_kmer_id_list)
        if len(seq_kmer) > max_len:
            max_len = len(seq_kmer)

    origin_token_list = list()
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        origin_token_list.append(seq_id)

    print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
    print('sequences_residue', sequences[0:5])  # sequences_residue
    print('token_list', token_list[0:5])
    print('len(token_list)', len(token_list))
    print('len(origin_token_list)', len(origin_token_list))
    print('new_token_list', new_token_list)
    print('num_token2index', num_token2index)
    print('len(token2index)', len(token2index))

    # update token2index
    with open('../data/kmer_residue2idx.pkl', 'wb') as file:
        pickle.dump(token2index, file)

    return token_list, origin_token_list, max_len


def make_data_with_unified_length(token_list, origin_token_list, labels, config):
    max_len = config.max_len + 2  # add [CLS] and [SEP]
    token2idx = config.token2index
    k_mer = config.k_mer
    kmer_num = (k_mer + 1) * k_mer // 2

    data = []
    for i in range(len(labels)):
        token_list[i] = [[token2idx['[CLS]']] * kmer_num] + token_list[i] + [[token2idx['[SEP]']] * kmer_num]
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([[0] * kmer_num] * n_pad)

        origin_token_list[i] = [token2idx['[CLS]']] + origin_token_list[i] + [token2idx['[SEP]']]
        n_pad = max_len - len(origin_token_list[i])
        origin_token_list[i].extend([token2idx['[PAD]']] * n_pad)

        data.append([token_list[i], origin_token_list[i], labels[i]])
        # print('token_list[i]', len(token_list[i]), token_list[i])
        # print('origin_token_list[i]', len(origin_token_list[i]), origin_token_list[i])

    return data


def construct_dataset(data, config):
    cuda = config.cuda
    batch_size = config.batch_size

    print('-' * 20, '[construct_dataset]: check data dimension', '-' * 20)
    print('len(data)', len(data))
    print('len(data[0])', len(data[0]))
    print('len(data[0][0])', len(data[0][0]))
    print('data[0][1]', data[0][1])
    print('len(data[1][0])', len(data[1][0]))
    print('data[1][1]', data[1][1])

    input_ids, origin_input_ids, labels = zip(*data)

    if cuda:
        input_ids, origin_input_ids, labels = torch.cuda.LongTensor(input_ids), torch.cuda.LongTensor(
            origin_input_ids), torch.cuda.LongTensor(labels)
    else:
        input_ids, origin_input_ids, labels = torch.LongTensor(input_ids), torch.LongTensor(
            origin_input_ids), torch.LongTensor(labels)

    print('-' * 20, '[construct_dataset]: check GPU data', '-' * 20)
    print('input_ids.device:', input_ids.device)
    print('origin_input_ids.device:', origin_input_ids.device)
    print('labels.device:', labels.device)

    print('-' * 20, '[construct_dataset]: check data shape', '-' * 20)
    print('input_ids:', input_ids.shape)  # [num_train_sequences, seq_len, kmer_num]
    print('origin_input_ids:', origin_input_ids.shape)  # [num_train_sequences, seq_len]
    print('labels:', labels.shape)  # [num_train_sequences, seq_len]

    data_loader = Data.DataLoader(MyDataSet(input_ids, origin_input_ids, labels),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False)

    print('len(data_loader)', len(data_loader))
    return data_loader


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, origin_input_ids, labels):
        self.input_ids = input_ids
        self.origin_input_ids = origin_input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.origin_input_ids[idx], self.labels[idx]


def load_data(config):
    path_data_train = config.path_train_data
    path_data_test = config.path_test_data

    sequences_train, labels_train = util_file.load_tsv_format_data(path_data_train)
    sequences_test, labels_test = util_file.load_tsv_format_data(path_data_test)

    token_list_train, origin_token_list_train, max_len_train = transform_token2index(sequences_train, config)
    token_list_test, origin_token_list_test, max_len_test = transform_token2index(sequences_test, config)
    config.max_len_train = max_len_train
    config.max_len_test = max_len_test
    config.max_len = max(max_len_train, max_len_test)

    data_train = make_data_with_unified_length(token_list_train, origin_token_list_train, labels_train, config)
    data_test = make_data_with_unified_length(token_list_test, origin_token_list_test, labels_test, config)

    data_loader_train = construct_dataset(data_train, config)
    data_loader_test = construct_dataset(data_test, config)

    return data_loader_train, data_loader_test


if __name__ == '__main__':
    '''
    check loading tsv data
    '''
    config = config.get_train_config()

    token2index = pickle.load(open('../data/kmer_residue2idx.pkl', 'rb'))
    config.token2index = token2index

    config.path_train_data = '../data/ACP_dataset/tsv/ACP_mixed_train.tsv'
    sequences, labels = util_file.load_tsv_format_data(config.path_train_data)
    token_list, origin_token_list, max_len = transform_token2index(sequences, config)
    data = make_data_with_unified_length(token_list, origin_token_list, labels, config)
    data_loader = construct_dataset(data, config)

    print('-' * 20, '[data_loader]: check data batch', '-' * 20)
    for i, batch in enumerate(data_loader):
        input, origin_input, label = batch
        print('batch[{}], input:{}, origin_input:{}, label:{}'.format(i, input.shape, origin_input.shape, label.shape))
