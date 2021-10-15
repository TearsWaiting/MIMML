import os
import pickle
import random
import torch
import torch.utils.data as Data
from util import util_file


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def read_meta_dataset_from_dir(path_meta_dataset):
    peptide_data_path_list = []
    for root, dirs, files in os.walk(path_meta_dataset):
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
                    # print('file[Pos]', file)
                    peptide_data_path[1] = file
                elif '(Neg)' in file:
                    # print('file[Neg]', file)
                    peptide_data_path[2] = file

        peptide_data_path_list.append(peptide_data_path)
        # print('peptide_data_path', peptide_data_path)
        # print('=' * 100)

    # print('peptide_data_path_list', peptide_data_path_list)
    peptide_data_path_list = peptide_data_path_list[1:]
    # print('peptide_data_path_list', len(peptide_data_path_list), peptide_data_path_list)
    # print('=' * 100)

    peptide_data_list = []
    for peptide_data_path in peptide_data_path_list:
        peptide_name = peptide_data_path[0]

        '''划分随机序列'''
        if peptide_name == 'Random Sequence':
            sequences, labels = util_file.read_tsv_data(peptide_data_path[1])
            shuffle(sequences)

            # 数量太多
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
                sequences, labels = util_file.read_tsv_data(data_pos_path)
                peptide_data[1] = [sequences, labels]

            if data_neg_path is not None:
                sequences, labels = util_file.read_tsv_data(data_neg_path)
                peptide_data[2] = [sequences, labels]

            peptide_data_list.append(peptide_data)

    '''构造各类多肽的数据集'''
    peptide_dataset = []
    for i, peptide in enumerate(peptide_data_list):
        peptide_name = peptide[0]
        class_data = peptide[1][0]
        class_data = [seq.upper() for seq in class_data]

        '''删除包含非法字符的序列'''
        invalid_residue = ['J', 'O', ' ']
        for residue in invalid_residue:
            for seq in class_data:
                if residue in seq:
                    print('Invalid Sequence [{}] in: {}, Remove'.format(seq, peptide_name))
                    class_data.remove(seq)

        class_label = [i for x in range(len(class_data))]
        peptide_dataset.append([peptide_name, class_data, class_label])
    print('peptide_dataset', len(peptide_dataset))

    '''查看统计数据'''
    for i, peptide_data in enumerate(peptide_dataset):
        peptide_name = peptide_data[0]
        num_pos_data = len(peptide_data[1])
        if peptide_data[2] is not None:
            num_neg_data = len(peptide_data[2])
        else:
            num_neg_data = None
        print('[{}] {} | {} | {}'.format(i, peptide_name, num_pos_data, num_neg_data))

    print('peptide_dataset[{}]'.format(len(peptide_dataset)))
    return peptide_dataset


def load_token2index(path_token2index):
    token2index = pickle.load(open(path_token2index, 'rb'))
    return token2index


def token2index(token2index, seq_list):
    index_list = []
    max_len = 0
    for seq in seq_list:
        seq_index = [token2index[token] for token in seq]
        index_list.append(seq_index)
        if len(seq) > max_len:
            max_len = len(seq)
    return index_list, max_len


def unify_length(id_list, dict_token2index, max_len):
    for i in range(len(id_list)):
        id_list[i] = [dict_token2index['[CLS]']] + id_list[i] + [dict_token2index['[SEP]']]
        n_pad = max_len - len(id_list[i]) + 2
        id_list[i].extend([0] * n_pad)
    return id_list


def construct_dataset(data, label, cuda=True):
    if cuda:
        input_ids, labels = torch.cuda.LongTensor(data), torch.cuda.LongTensor(label)
    else:
        input_ids, labels = torch.LongTensor(data), torch.LongTensor(label)
    dataset = MyDataSet(input_ids, labels)
    return dataset


def construct_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    data_loader = Data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  drop_last=drop_last)
    return data_loader


def construct_BPD36_dataloader(path_dataset, path_token2index, model_max_len, batch_size, device):
    raw_peptide_dataset = read_meta_dataset_from_dir(path_dataset)
    dict_token2index = load_token2index(path_token2index)

    data_max_len = 0
    label_dict = {}
    all_seq_list = []
    all_label_list = []
    meta_data_ids = []
    meta_unified_ids = []
    for i, peptide in enumerate(raw_peptide_dataset):
        # raw_peptide_dataset: [peptide_name, seq_list, label_list]
        label_dict[i] = peptide[0]
        all_label_list.extend(peptide[2])
        class_i_seq_list = peptide[1]

        class_i_data_ids, class_i_max_len = token2index(dict_token2index, class_i_seq_list)
        data_max_len = max(data_max_len, class_i_max_len)
        meta_data_ids.append(class_i_data_ids)

    max_len = max(model_max_len, data_max_len)
    for class_i_data_ids in meta_data_ids:
        class_i_unified_ids = unify_length(class_i_data_ids, dict_token2index, max_len)
        meta_unified_ids.append(class_i_unified_ids)
        all_seq_list.extend(class_i_unified_ids)

    print('=' * 100)
    print('max_len', max_len)
    print('data_max_len', data_max_len)
    print('model_max_len', model_max_len)
    print('self.label_dict', label_dict)
    print('=' * 100)
    # print('len(meta_unified_ids)', len(meta_unified_ids))
    # print('all_seq_list', len(all_seq_list), all_seq_list[:20])
    # print('all_label_list', len(all_label_list), all_label_list[:20])

    filtered_seqs = []
    filtered_labels = []
    for i in range(len(all_label_list)):
        if all_label_list[i] < 28:
            filtered_seqs.append(all_seq_list[i])
            filtered_labels.append(all_label_list[i])

    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(filtered_seqs)
    random.seed(randnum)
    random.shuffle(filtered_labels)

    train_seqs = filtered_seqs[:int(0.8 * len(filtered_seqs))]
    train_labels = filtered_labels[:int(0.8 * len(filtered_labels))]
    test_seqs = filtered_seqs[int(0.8 * len(filtered_seqs)):]
    test_labels = filtered_labels[int(0.8 * len(filtered_labels)):]

    if device == 'cpu':
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(device)
    dataset_train = construct_dataset(train_seqs, train_labels, cuda)
    dataset_test = construct_dataset(test_seqs, test_labels, cuda)
    print('dataset_train', len(dataset_train), dataset_train)
    print('dataset_test', len(dataset_test), dataset_test)

    dataloader_train = construct_dataloader(dataset_train, batch_size)
    dataloader_test = construct_dataloader(dataset_test, batch_size)
    print('dataloader_train', len(dataloader_train), dataloader_train)
    print('dataloader_test', len(dataloader_test), dataloader_test)
    return dataloader_train, dataloader_test


if __name__ == '__main__':
    path_dataset = '../data/task_data/BPD-36'
    path_token2index = '../data/meta_data/residue2idx.pkl'
    BPD36_loader_train, BPD36_loader_test = construct_BPD36_dataloader(path_dataset, path_token2index, 207, 320, 0)

    for i, batch in enumerate(BPD36_loader_train):
        data, label = batch
        print('[{}] label:{}'.format(i, label))
