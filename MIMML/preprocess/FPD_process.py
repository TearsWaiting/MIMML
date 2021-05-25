# ---encoding:utf-8---
# @Time : 2021.03.18
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : FPD_process.py


import torch.utils.data as Data
import learn2learn as l2l
import pickle
import torch
import os

from random import shuffle
from configuration import config as configur
from util import util_file

'''修改dataset'''


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def transform_token2index(sequences, config):
    token2index = config.token2index

    sequences_residue_list = []
    for i, seq in enumerate(sequences):
        sequences_residue_list.append(list(seq))

    token_list = list()
    max_len = 0

    for i, seq in enumerate(sequences_residue_list):
        # print('[{}]: {}'.format(i, sequences[i]))
        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)

    return token_list, max_len


def make_data_with_unified_length(token_list, config):
    padded_max_len = config.max_len = config.max_len + 2  # add [CLS] and [SEP]
    token2index = config.token2index

    data = []
    for i in range(len(token_list)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]
        n_pad = padded_max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append(token_list[i])

    return data, padded_max_len


def transform2index(data, config):
    token_list, sequences_max_len = transform_token2index(data, config)
    print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
    print('len(token_list)', len(token_list))
    print('sequences_max_len', sequences_max_len)
    config.max_len = sequences_max_len
    data_index, padded_max_len = make_data_with_unified_length(token_list, config)
    print('-' * 20, '[make_data_with_unified_length]: check token_list head', '-' * 20)
    print('len(data_index)', len(data_index))
    print('padded_max_len', padded_max_len)
    return data_index


def get_FPD_dataset(config):
    # config = configur.get_train_config()
    token2index = pickle.load(open('../data/meta_data/residue2idx.pkl', 'rb'))
    config.token2index = token2index
    config.vocab_size = len(token2index)
    print('config.token2index', config.token2index)
    print('config.vocab_size', config.vocab_size)

    '''获取原始数据'''
    peptide_data_pathset_filepath = []
    # data_dir = '../data/task_data/Functional Peptides'
    data_dir = '../data/task_data/FPD'
    for root, dirs, files in os.walk(data_dir):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        peptide_data_path = [None, None, None]
        peptide_name = root.split('/')[-1]
        peptide_data_path[0] = peptide_name

        # print('root', root)
        # print('peptide_name', peptide_name)

        # 遍历文件
        for f in files:
            file = os.path.join(root, f)
            if file.endswith('.tsv'):
                if '(Pos)' in file:
                    print('file[Pos]', file)
                    peptide_data_path[1] = file
                elif '(Neg)' in file:
                    print('file[Neg]', file)
                    peptide_data_path[2] = file

        peptide_data_pathset_filepath.append(peptide_data_path)
        print('peptide_data_path', peptide_data_path)
        print('=' * 100)

    print('peptide_data_pathset_filepath', peptide_data_pathset_filepath)
    peptide_data_pathset_filepath = peptide_data_pathset_filepath[1:]
    print('peptide_data_pathset_filepath', peptide_data_pathset_filepath)
    print('=' * 100)

    peptide_data_list = []
    for peptide_data_path in peptide_data_pathset_filepath:
        peptide_name = peptide_data_path[0]

        '''划分随机序列'''
        if peptide_name == 'Random Sequence':
            data_pos_path = peptide_data_path[1]
            sequences, labels = util_file.load_tsv_format_data(data_pos_path)
            shuffle(sequences)

            # 数量太多
            # random_sequence_data_meta_train = sequences[:len(sequences) // 2]
            # random_sequence_data_meta_test = sequences[len(sequences) // 2:]
            # labels_data_meta_train = labels[:len(sequences) // 2]
            # labels_data_meta_test = labels[len(sequences) // 2:]

            random_sequence_data_meta_train = sequences[:1000]
            random_sequence_data_meta_test = sequences[1000:2000]
            labels_data_meta_train = labels[:1000]
            labels_data_meta_test = labels[1000:2000]

            peptide_data = ['Random Sequence Meta Train',
                            [random_sequence_data_meta_train, labels_data_meta_train], None]
            peptide_data_list.append(peptide_data)
            peptide_data = ['Random Sequence Meta Test',
                            [random_sequence_data_meta_test, labels_data_meta_test], None]
            peptide_data_list.append(peptide_data)
        else:
            peptide_data = [peptide_name, None, None]

            data_pos_path = peptide_data_path[1]
            data_neg_path = peptide_data_path[2]
            if data_pos_path is not None:
                sequences, labels = util_file.load_tsv_format_data(data_pos_path)
                peptide_data[1] = [sequences, labels]

            if data_neg_path is not None:
                sequences, labels = util_file.load_tsv_format_data(data_neg_path)
                peptide_data[2] = [sequences, labels]

            # 数量太多
            if peptide_name == 'MHC Class I Binding Peptides':
                sequences = sequences[:3000]
                labels = labels[:3000]
                peptide_data[1] = [sequences, labels]

            peptide_data_list.append(peptide_data)

    print('=' * 100)

    '''查看统计数据'''
    for i, peptide_data in enumerate(peptide_data_list):
        peptide_name = peptide_data[0]
        num_pos_data = len(peptide_data[1][0])
        if peptide_data[2] is not None:
            num_neg_data = len(peptide_data[2][0])
        else:
            num_neg_data = None
        print('[{}] {} | {} | {}'.format(i, peptide_name, num_pos_data, num_neg_data))

    '''
    peptide_data_list: [num_type_of_peptide = 22, 3]
    peptide_data_list[i][0]: peptide_name
    peptide_data_list[i][1]: peptide_pos_data, 
        peptide_data_list[i][1][0]: sequnces
        peptide_data_list[i][1][1]: labels
    peptide_data_list[i][2]: peptide_neg_data,
        peptide_data_list[i][2][0]: sequnces
        peptide_data_list[i][2][1]: labels
    '''

    '''构造各类多肽的数据集'''
    peptide_dataset = []
    for i, peptide in enumerate(peptide_data_list):
        peptide_name = peptide[0]
        class_data = peptide[1][0]

        '''删除非法序列'''
        invalid_seqs = ['JGLPPGPPIPP', 'ACEINHIBITOR']
        # invalid_seqs = ['ACEINHIBITOR']
        for invalid_seq in invalid_seqs:
            if invalid_seq in class_data:
                print('Invalid Sequence [{}] in: {}, Remove'.format(invalid_seq, peptide_name))
                class_data.remove(invalid_seq)

        class_label = [i for x in range(len(class_data))]
        peptide_dataset.append([peptide_name, class_data, class_label])

    '''构建torch.Dataset'''
    data = []
    label = []
    label_dict = {}
    for i, peptide in enumerate(peptide_dataset):
        data.extend(peptide[1])
        label.extend(peptide[2])
        label_dict[i] = peptide[0]

    print('=' * 100)
    print('len(data)', len(data))
    print('len(label)', len(label))
    print('label_dict', label_dict)

    '''将字母序列转换为token列表'''
    data = transform2index(data, config)  # data: [num_samples]

    print('-' * 100)
    for i in range(5):
        print('data[{}]:{}'.format(i, data[i]))
    print('-' * 100)

    FPD_dataset = MyDataSet(data, label)
    print('FPD_dataset', FPD_dataset)
    return FPD_dataset


def test_meta_dataset():
    config = configur.get_train_config()
    token2index = pickle.load(open('../data/meta_data/residue2idx.pkl', 'rb'))
    config.token2index = token2index
    config.vocab_size = len(token2index)
    print('config.token2index', config.token2index)
    print('config.vocab_size', config.vocab_size)

    '''获取原始数据'''
    peptide_data_pathset_filepath = []
    # data_dir = '../data/task_data/Functional Peptides'
    data_dir = '../data/task_data/FPD'
    for root, dirs, files in os.walk(data_dir):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        peptide_data_path = [None, None, None]
        peptide_name = root.split('/')[-1]
        peptide_data_path[0] = peptide_name

        # print('root', root)
        # print('peptide_name', peptide_name)

        # 遍历文件
        for f in files:
            file = os.path.join(root, f)
            if file.endswith('.tsv'):
                if '(Pos)' in file:
                    print('file[Pos]', file)
                    peptide_data_path[1] = file
                elif '(Neg)' in file:
                    print('file[Neg]', file)
                    peptide_data_path[2] = file

        peptide_data_pathset_filepath.append(peptide_data_path)
        print('peptide_data_path', peptide_data_path)
        print('=' * 100)

    print('peptide_data_pathset_filepath', peptide_data_pathset_filepath)
    peptide_data_pathset_filepath = peptide_data_pathset_filepath[1:]
    print('peptide_data_pathset_filepath', peptide_data_pathset_filepath)
    print('=' * 100)

    peptide_data_list = []
    for peptide_data_path in peptide_data_pathset_filepath:
        peptide_name = peptide_data_path[0]

        if peptide_name == 'Random Sequence':
            data_pos_path = peptide_data_path[1]
            sequences, labels = util_file.load_tsv_format_data(data_pos_path)
            shuffle(sequences)
            random_sequence_data_meta_train = sequences[:len(sequences) // 2]
            random_sequence_data_meta_test = sequences[len(sequences) // 2:]
            labels_data_meta_train = sequences[:len(sequences) // 2]
            labels_data_meta_test = sequences[len(sequences) // 2:]
            peptide_data = ['Random Sequence Meta Train',
                            [random_sequence_data_meta_train, labels_data_meta_train], None]
            peptide_data_list.append(peptide_data)
            peptide_data = ['Random Sequence Meta Test',
                            [random_sequence_data_meta_test, labels_data_meta_test], None]
            peptide_data_list.append(peptide_data)
        else:
            peptide_data = [peptide_name, None, None]

            data_pos_path = peptide_data_path[1]
            data_neg_path = peptide_data_path[2]
            if data_pos_path is not None:
                sequences, labels = util_file.load_tsv_format_data(data_pos_path)
                peptide_data[1] = [sequences, labels]

            if data_neg_path is not None:
                sequences, labels = util_file.load_tsv_format_data(data_neg_path)
                peptide_data[2] = [sequences, labels]
            peptide_data_list.append(peptide_data)

    print('=' * 100)

    '''查看统计数据'''
    for i, peptide_data in enumerate(peptide_data_list):
        peptide_name = peptide_data[0]
        num_pos_data = len(peptide_data[1][0])
        if peptide_data[2] is not None:
            num_neg_data = len(peptide_data[2][0])
        else:
            num_neg_data = None
        print('[{}] {} | {} | {}'.format(i, peptide_name, num_pos_data, num_neg_data))

    '''
    peptide_data_list: [num_type_of_peptide = 22, 3]
    peptide_data_list[i][0]: peptide_name
    peptide_data_list[i][1]: peptide_pos_data, 
        peptide_data_list[i][1][0]: sequnces
        peptide_data_list[i][1][1]: labels
    peptide_data_list[i][2]: peptide_neg_data,
        peptide_data_list[i][2][0]: sequnces
        peptide_data_list[i][2][1]: labels
    '''

    '''构造各类多肽的数据集'''
    peptide_dataset = []
    for i, peptide in enumerate(peptide_data_list):
        peptide_name = peptide[0]
        class_data = peptide[1][0]

        '''删除非法序列'''
        invalid_seqs = ['JGLPPGPPIPP', 'ACEINHIBITOR']
        for invalid_seq in invalid_seqs:
            if invalid_seq in class_data:
                print('Invalid Sequence in: {}'.format(peptide_name), 'Remove')
                class_data.remove(invalid_seq)

        class_label = [i for x in range(len(class_data))]
        peptide_dataset.append([peptide_name, class_data, class_label])

    '''构建torch.Dataset'''
    data = []
    label = []
    label_dict = {}
    for i, peptide in enumerate(peptide_dataset):
        data.extend(peptide[1])
        label.extend(peptide[2])
        label_dict[i] = peptide[0]

    print('=' * 100)
    print('len(data)', len(data))
    print('len(label)', len(label))
    print('label_dict', label_dict)

    '''将字母序列转换为token列表'''
    print('=' * 100)
    print('raw data sequences', data[:20])
    print('label', label[:20])
    data = transform2index(data, config)  # data: [num_samples]

    print('-' * 100)
    for i in range(5):
        print('data[{}] label[{}]:{}'.format(i, label[i], data[i]))
    print('-' * 100)

    FPD_dataset = MyDataSet(data, label)
    print('FPD_dataset', FPD_dataset)

    print('-' * 100)
    for i in range(5):
        print('FPD_dataset[{}]:{}'.format(i, FPD_dataset[i]))
    print('-' * 100)

    '''构建MetaDataset'''
    # FPD_dataset = l2l.data.MetaDataset(FPD_dataset)
    FPD_dataset = l2l.data.FilteredMetaDataset(FPD_dataset, [0, 1])
    print('FPD_dataset', FPD_dataset)
    print('len(FPD_dataset)', len(FPD_dataset), type(FPD_dataset))

    print('-' * 100)
    for i in range(5):
        print('MetaDataset[{}]:{}'.format(i, FPD_dataset[i]))
    print('-' * 100)

    FDP_tasks = l2l.data.TaskDataset(
        dataset=FPD_dataset,
        task_transforms=[
            l2l.data.transforms.NWays(FPD_dataset, n=2),
            l2l.data.transforms.KShots(FPD_dataset, k=5 + 15),
            l2l.data.transforms.LoadData(FPD_dataset),
        ],
        num_tasks=20000,
    )
    print('FDP_tasks', FDP_tasks)

    '''Sampling'''
    task = FDP_tasks.sample()
    data, labels = task
    print('data', len(data), data)
    print('labels', labels.size(), labels)

    print('=' * 100)

    '''检查tokens是否正确地对应序列'''
    index2token = {}
    for key, value in token2index.items():
        index2token[value] = key
    print('index2token', index2token)

    data = [x.view(-1, 1) for x in data]
    data = torch.cat(data, dim=1)
    print('data', data.size())
    print('data', data)

    for tensor in data:
        seq = ''
        for x in tensor:
            x = int(x)
            if x >= 4:
                seq += index2token[x]
        print('seq', seq)


def split_train_test():
    config = configur.get_train_config()
    token2index = pickle.load(open('../data/meta_data/residue2idx.pkl', 'rb'))
    config.token2index = token2index
    config.vocab_size = len(token2index)
    print('config.token2index', config.token2index)
    print('config.vocab_size', config.vocab_size)

    '''获取原始数据'''
    peptide_data_pathset_filepath = []
    # data_dir = '../data/task_data/Functional Peptides'
    data_dir = '../data/task_data/FPD'
    for root, dirs, files in os.walk(data_dir):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        peptide_data_path = [None, None, None]
        peptide_name = root.split('/')[-1]
        peptide_data_path[0] = peptide_name

        # print('root', root)
        # print('peptide_name', peptide_name)

        # 遍历文件
        for f in files:
            file = os.path.join(root, f)
            if file.endswith('.tsv'):
                if '(Pos)' in file:
                    print('file[Pos]', file)
                    peptide_data_path[1] = file
                elif '(Neg)' in file:
                    print('file[Neg]', file)
                    peptide_data_path[2] = file

        peptide_data_pathset_filepath.append(peptide_data_path)
        print('peptide_data_path', peptide_data_path)
        print('=' * 100)

    print('peptide_data_pathset_filepath', peptide_data_pathset_filepath)
    peptide_data_pathset_filepath = peptide_data_pathset_filepath[1:]
    print('peptide_data_pathset_filepath', peptide_data_pathset_filepath)
    print('=' * 100)

    peptide_data_list = []
    for peptide_data_path in peptide_data_pathset_filepath:
        peptide_name = peptide_data_path[0]

        if peptide_name == 'Random Sequence':
            data_pos_path = peptide_data_path[1]
            sequences, labels = util_file.load_tsv_format_data(data_pos_path)
            shuffle(sequences)
            random_sequence_data_meta_train = sequences[:len(sequences) // 2]
            random_sequence_data_meta_test = sequences[len(sequences) // 2:]
            labels_data_meta_train = sequences[:len(sequences) // 2]
            labels_data_meta_test = sequences[len(sequences) // 2:]
            peptide_data = ['Random Sequence Meta Train',
                            [random_sequence_data_meta_train, labels_data_meta_train], None]
            peptide_data_list.append(peptide_data)
            peptide_data = ['Random Sequence Meta Test',
                            [random_sequence_data_meta_test, labels_data_meta_test], None]
            peptide_data_list.append(peptide_data)
        else:
            peptide_data = [peptide_name, None, None]

            data_pos_path = peptide_data_path[1]
            data_neg_path = peptide_data_path[2]
            if data_pos_path is not None:
                sequences, labels = util_file.load_tsv_format_data(data_pos_path)
                peptide_data[1] = [sequences, labels]

            if data_neg_path is not None:
                sequences, labels = util_file.load_tsv_format_data(data_neg_path)
                peptide_data[2] = [sequences, labels]
            peptide_data_list.append(peptide_data)

    print('=' * 100)

    '''查看统计数据'''
    for i, peptide_data in enumerate(peptide_data_list):
        peptide_name = peptide_data[0]
        num_pos_data = len(peptide_data[1][0])
        if peptide_data[2] is not None:
            num_neg_data = len(peptide_data[2][0])
        else:
            num_neg_data = None
        print('[{}] {} | {} | {}'.format(i, peptide_name, num_pos_data, num_neg_data))

    '''
    peptide_data_list: [num_type_of_peptide = 22, 3]
    peptide_data_list[i][0]: peptide_name
    peptide_data_list[i][1]: peptide_pos_data, 
        peptide_data_list[i][1][0]: sequnces
        peptide_data_list[i][1][1]: labels
    peptide_data_list[i][2]: peptide_neg_data,
        peptide_data_list[i][2][0]: sequnces
        peptide_data_list[i][2][1]: labels
    '''

    for peptide_data in peptide_data_list:
        # 如果该类别有负样本
        if peptide_data[2] is not None:
            peptide_name = peptide_data[0]
            pos_samples = peptide_data[1][0]
            neg_samples = peptide_data[2][0]

            shuffle(pos_samples)
            shuffle(neg_samples)

            pos_length = len(pos_samples)
            neg_length = len(neg_samples)
            selected_len = min(pos_length, neg_length)
            split_rate = 0.8
            threshold = int(selected_len * split_rate)

            train_pos_samples = pos_samples[:threshold]
            train_neg_samples = neg_samples[:threshold]
            test_pos_samples = pos_samples[threshold:selected_len]
            test_neg_samples = neg_samples[threshold:selected_len]

            train_samples = train_pos_samples + train_neg_samples
            test_samples = test_pos_samples + test_neg_samples
            train_labels = [1 for i in range(len(train_pos_samples))] + [0 for i in range(len(train_neg_samples))]
            test_labels = [1 for i in range(len(test_pos_samples))] + [0 for i in range(len(test_neg_samples))]
            tsv_filename = '../data/task_data/FDP/' + peptide_name + ' train.tsv'
            util_file.write_tsv_format_data(tsv_filename, train_labels, train_samples)
            tsv_filename = '../data/task_data/FDP/' + peptide_name + ' test.tsv'
            util_file.write_tsv_format_data(tsv_filename, test_labels, test_samples)


if __name__ == '__main__':
    test_meta_dataset()
    # split_train_test()
