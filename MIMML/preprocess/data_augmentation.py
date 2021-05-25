# ---encoding:utf-8---
# @Time : 2021.03.22
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : data_augmentation.py

from util import util_file
from random import random, randint
from configuration import config

import pickle


def get_reverse(seq, keep_rate=0.8):
    if random() >= keep_rate:
        seq = list(seq)
        seq.reverse()
        seq = ''.join(seq)

    return seq


def get_random_replacement(seq, config, keep_rate=0.8):
    token2index = config.token2index
    token_list = list(token2index.keys())
    rand_replace = lambda c: c if random() < keep_rate else token_list[randint(4, len(token_list) - 1)]
    seq_rand_replace = ''.join([rand_replace(c) for c in seq])
    return seq_rand_replace


def augmentation(path_tsv_data, config, append=True):
    sequences, labels = util_file.load_tsv_format_data(path_tsv_data)
    print('sequences', len(sequences))
    print('labels', len(labels))

    sequences_augment = []
    labels_augment = []
    for i in range(len(labels)):
        seq = sequences[i]
        label = labels[i]

        # reverse seqeunce order
        seq_reverse = get_reverse(seq, keep_rate = 0.8)
        print('seq', seq)
        print('seq_reverse', seq_reverse)

        # random replacement
        # seq_rand_replace = get_random_replacement(seq, config, keep_rate=0.5)
        # print('seq', seq)
        # print('seq_rand_replace', seq_rand_replace)

        sequences_augment.append(seq_reverse)
        labels_augment.append(label)
        # sequences_augment.append(seq_rand_replace)
        # labels_augment.append(label)

    if append:
        sequences = sequences + sequences_augment
        labels = labels + labels_augment
    else:
        sequences = sequences_augment
        labels = labels_augment

    return sequences, labels


if __name__ == '__main__':
    path_train_data = '../data/ACP_dataset/tsv/ACP-Mixed-80-train.tsv'
    path_test_data = '../data/ACP_dataset/tsv/ACP-Mixed-80-test.tsv'
    config = config.get_train_config()
    token2index = pickle.load(open('../data/residue2idx.pkl', 'rb'))
    config.token2index = token2index
    print('token2index', token2index)
    sequences, labels = augmentation(path_train_data, config, append=False)
    print('sequences', len(sequences))
    print('labels', len(labels))
