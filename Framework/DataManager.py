import os
import pickle
import torch
import torch.utils.data as Data
import learn2learn as l2l
import random
import numpy as np
import copy
from util import util_file
from random import shuffle
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.datasets.helpers import miniimagenet


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class DataManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.config = learner.config

        self.mode = self.config.mode
        self.token2index = None
        self.data_max_len = 0

        if self.config.cuda:
            self.device = torch.device('cuda')
            torch.cuda.set_device(self.config.device)
            if self.config.seed:
                torch.cuda.manual_seed(self.config.seed)
        else:
            self.device = torch.device('cpu')

        # label:
        self.train_label = None
        self.valid_label = None
        self.test_label = None
        # raw_data: ['MNH', 'APD', ...]
        self.train_raw_data = None
        self.valid_raw_ata = None
        self.test_raw_data = None
        # data_ids: [[1, 5, 8], [2, 7, 9], ...]
        self.train_data_ids = None
        self.valid_data_ids = None
        self.test_data_ids = None
        # data_ids: [[1, 5, 8, '[PAD]', ..., '[PAD]'], [2, 7, 9, '[PAD]', ..., '[PAD]'], ...]
        self.train_unified_ids = None
        self.valid_unified_ids = None
        self.test_unified_ids = None
        # dataset
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        # iterator
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        # meta dataset
        self.label_dict = {}
        self.class_label = []
        self.meta_raw_data = []
        self.meta_data_ids = []
        self.meta_unified_ids = []
        self.meta_dataset = None
        self.meta_dataloader = None
        self.meta_train_class = None
        self.meta_valid_class = None
        self.meta_test_class = None

        # miniImageNet
        self.miniImageNet_dataloader = []
        self.miniImageNet_generator = []

        # multiple classification
        self.multiple_classification_dataset = None

    def load_data(self):
        self.__load_token2index()
        self.__load_raw_data()
        self.__process_data()

        # set max length for model initialization
        print('Final Max Length: {} (config.max_len: {}, data_max_len:{})'.format(
            max(self.config.max_len, self.data_max_len), self.config.max_len, self.data_max_len))
        if self.config.max_len < self.data_max_len:
            self.config.max_len = self.data_max_len

    def get_dataloder(self, name):
        return_data = None
        if name == 'train_set':
            return_data = self.train_dataloader
        elif name == 'valid_set':
            return_data = self.valid_dataloader
        elif name == 'test_set':
            return_data = self.test_dataloader
        elif name == 'meta_train':
            if self.config.dataset == 'Peptide Sequence':
                return_data = self.meta_dataloader[0]
            elif self.config.dataset == 'miniImageNet':
                return_data = self.miniImageNet_generator[0]
            else:
                self.IOManager.log.Error('No Such Dataset')
        elif name == 'meta_valid':
            if self.config.dataset == 'Peptide Sequence':
                return_data = self.meta_dataloader[1]
            elif self.config.dataset == 'miniImageNet':
                return_data = self.miniImageNet_generator[1]
            else:
                self.IOManager.log.Error('No Such Dataset')
        elif name == 'meta_test':
            if self.config.dataset == 'Peptide Sequence':
                return_data = self.meta_dataloader[2]
            elif self.config.dataset == 'miniImageNet':
                return_data = self.miniImageNet_generator[2]
            else:
                self.IOManager.log.Error('No Such Dataset')
        else:
            self.IOManager.log.Error('No Such Name')
        return return_data

    def sample_task(self, tasks):
        if self.config.dataset == 'Peptide Sequence':
            return tasks.sample()
        elif self.config.dataset == 'miniImageNet':
            try:
                batch = next(tasks)
            except StopIteration:
                return None

            support_inputs, support_targets = batch['train']
            query_inputs, query_targets = batch['test']
            if self.config.cuda:
                support_inputs = support_inputs.to(device=self.device)
                support_targets = support_targets.to(device=self.device)
                query_inputs = query_inputs.to(device=self.device)
                query_targets = query_targets.to(device=self.device)
                # support_inputs: [task_batch_size, support_num_samples = n_ways * k_support_shots, channel, width, height]
                # support_targets: [task_batch_size, support_num_samples = n_ways * k_support_shots]
                # query_inputs: [task_batch_size, query_num_samples = n_ways * k_query_shots, channel, width, height]
                # query_targets: [task_batch_size, query_num_samples = n_ways * k_query_shots]
            return [support_inputs[0], support_targets[0], query_inputs[0], query_targets[0]]

    def get_inference_task(self):
        if self.config.dataset == 'imbalanced inference dataset':
            support_samples = self.support_samples
            support_labels = self.support_labels
            query_samples = self.query_samples
            query_labels = self.query_labels
        elif self.config.dataset == 'inference dataset':
            support_samples, support_labels = self.inference_support_set.sample()
            query_samples, query_labels = self.inference_query_set.sample()
            # query_samples = self.query_samples
            # query_labels = self.query_labels
        else:
            self.IOManager.log.Error('No Such Dataset')

        return [support_samples, support_labels, query_samples, query_labels]

    def reload_iterator(self):
        self.miniImageNet_generator[1] = self.__batch_task_generator(self.miniImageNet_dataloader[1])
        return self.miniImageNet_generator[1]

    def __load_raw_data(self):
        '''
        mode == 'train-test': read 'train_data.tsv' and 'test_data.tsv' file
        mode == 'cross validation': read 'train_data.tsv' file
        mode == 'meta-data': read files in a folder
        '''
        if self.mode == 'train-test':
            if self.config.num_class == 2:
                self.train_raw_data, self.train_label = util_file.read_tsv_data(self.config.path_train_data,
                                                                                skip_first=True)
                self.test_raw_data, self.test_label = util_file.read_tsv_data(self.config.path_test_data,
                                                                              skip_first=True)
            else:
                self.__read_meta_dataset_from_dir(self.config.path_dataset)
        elif self.mode == 'cross validation':
            self.train_raw_data, self.train_label = util_file.read_tsv_data(self.config.path_train_data,
                                                                            skip_first=True)
        elif self.mode == 'meta learning':
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
            if self.config.dataset == 'miniImageNet':
                pass
            elif self.config.dataset == 'Peptide Sequence':
                self.__read_meta_dataset_from_dir(self.config.path_meta_dataset)
            elif 'inference dataset' in self.config.dataset:
                self.train_raw_data, self.train_label = util_file.read_tsv_data(self.config.path_train_data,
                                                                                skip_first=True)
                self.test_raw_data, self.test_label = util_file.read_tsv_data(self.config.path_test_data,
                                                                              skip_first=True)
            else:
                self.IOManager.log.Error('No Such Dataset')
        else:
            self.IOManager.log.Error('No Such Mode')

    def __process_data(self):
        if self.mode == 'train-test':
            if self.config.num_class == 2:
                self.__process_binary_classification_dataset()
            else:
                self.__process_multiple_classification_dataset()
        elif self.mode == 'cross validation':
            self.__process_cross_validation_dataset()
        elif self.mode == 'meta learning':
            if self.config.dataset == 'miniImageNet':
                self.__process_miniImageNet()
            elif self.config.dataset == 'Peptide Sequence':
                self.__process_peptide_sequence()
            elif 'inference dataset' in self.config.dataset:
                self.__process_binary_classification_dataset()
            else:
                self.IOManager.log.Error('No Such Dataset')
        else:
            self.IOManager.log.Error('No Such Mode')

    def __load_token2index(self):
        self.token2index = pickle.load(open(self.config.path_token2index, 'rb'))
        self.config.vocab_size = len(self.token2index)

    def __token2index(self, token2index, seq_list):
        index_list = []
        max_len = 0
        for seq in seq_list:
            seq_index = [token2index[token] for token in seq]
            index_list.append(seq_index)
            if len(seq) > max_len:
                max_len = len(seq)
        return index_list, max_len

    def __unify_length(self, id_list, token2index, max_len):
        for i in range(len(id_list)):
            id_list[i] = [token2index['[CLS]']] + id_list[i] + [token2index['[SEP]']]
            n_pad = max_len - len(id_list[i]) + 2
            id_list[i].extend([0] * n_pad)
        return id_list

    def __construct_dataset(self, data, label, cuda=True):
        if cuda:
            input_ids, labels = torch.cuda.LongTensor(data), torch.cuda.LongTensor(label)
        else:
            input_ids, labels = torch.LongTensor(data), torch.LongTensor(label)
        dataset = MyDataSet(input_ids, labels)
        return dataset

    def __construct_dataloader(self, dataset, batch_size, shuffle=True, drop_last=False):
        data_loader = Data.DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      drop_last=drop_last)
        return data_loader

    def __read_meta_dataset_from_dir(self, path_dataset):
        peptide_data_path_list = []
        for root, dirs, files in os.walk(path_dataset):
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
                    peptide_data_path[1] = file

            peptide_data_path_list.append(peptide_data_path)

        peptide_data_path_list = peptide_data_path_list[1:]
        # print('peptide_data_path_list', len(peptide_data_path_list), peptide_data_path_list)
        # print('=' * 100)

        meta_dataset_random = []
        pretrain_dataset_random = []
        for peptide_data_path in peptide_data_path_list:
            peptide_name = peptide_data_path[0]

            if peptide_name == 'Random Sequence':
                sequences, labels = util_file.read_tsv_data(peptide_data_path[1])
                shuffle(sequences)

                # 数量太多
                random_sequence_data_meta_train = sequences[:1000]
                random_sequence_data_meta_valid = sequences[1000:2000]
                random_sequence_data_meta_test = sequences[2000:3000]

                peptide_data = ['Random Sequence for Meta Train',
                                random_sequence_data_meta_train, None]
                meta_dataset_random.append(peptide_data)
                peptide_data = ['Random Sequence for Meta Valid',
                                random_sequence_data_meta_valid, None]
                meta_dataset_random.append(peptide_data)
                peptide_data = ['Random Sequence for Meta Test',
                                random_sequence_data_meta_test, None]
                meta_dataset_random.append(peptide_data)

                shuffle(sequences)
                peptide_data = ['Random Sequence for Pretrain', sequences[:1000], None]
                pretrain_dataset_random.append(peptide_data)

        meta_dataset = []
        for peptide_data_path in peptide_data_path_list:
            if peptide_data_path[0] != 'Random Sequence':
                peptide_name = peptide_data_path[0]
                peptide_data = None
                data_pos_path = peptide_data_path[1]
                if data_pos_path is not None:
                    sequences, labels = util_file.read_tsv_data(data_pos_path)
                    peptide_data = [peptide_name, sequences, None]
                meta_dataset.append(peptide_data)

        random.seed(self.config.seed)
        random.shuffle(meta_dataset)

        pretrain_dataset = pretrain_dataset_random + copy.deepcopy(meta_dataset[:self.config.num_meta_train])
        meta_dataset = meta_dataset_random + meta_dataset

        '''分配标签'''
        for i in range(len(meta_dataset)):
            meta_dataset[i][2] = i

        for i in range(len(pretrain_dataset)):
            pretrain_dataset[i][2] = i

        '''查看统计数据'''
        print('=' * 50, 'meta dataset', '=' * 50)
        for class_data in meta_dataset:
            peptide_name = class_data[0]
            class_index = class_data[2]
            num_class_data = len(class_data[1])
            print('label[{}]: {}  | {} |'.format(class_index, peptide_name, num_class_data))

        # print('=' * 50, 'original pretrain dataset', '=' * 50)
        for class_data in pretrain_dataset:
            peptide_name = class_data[0]
            class_index = class_data[2]
            num_class_data = len(class_data[1])
            # print('label[{}]: {}  | {} |'.format(class_index, peptide_name, num_class_data))

        # 去除部分类别中数量太多的序列
        for i, class_data in enumerate(pretrain_dataset):
            if len(class_data[1]) > 1000:
                shuffle(class_data[1])
                pretrain_dataset[i][1] = class_data[1][:1000]

        # print('=' * 50, 'final pretrain dataset', '=' * 50)
        for class_data in pretrain_dataset:
            peptide_name = class_data[0]
            class_index = class_data[2]
            num_class_data = len(class_data[1])
            # print('label[{}]: {}  | {} |'.format(class_index, peptide_name, num_class_data))

        self.meta_raw_data = meta_dataset
        self.multiple_classification_dataset = pretrain_dataset

    def __process_binary_classification_dataset(self):
        '''token2index'''
        self.train_data_ids, train_max_len = self.__token2index(self.token2index, self.train_raw_data)
        self.test_data_ids, test_max_len = self.__token2index(self.token2index, self.test_raw_data)
        self.data_max_len = max(train_max_len, test_max_len)
        self.data_max_len = max(self.data_max_len, self.config.max_len)

        '''unify_length'''
        self.train_unified_ids = self.__unify_length(self.train_data_ids, self.token2index, self.data_max_len)
        self.test_unified_ids = self.__unify_length(self.test_data_ids, self.token2index, self.data_max_len)

        '''construct_dataset'''
        self.train_dataset = self.__construct_dataset(self.train_unified_ids, self.train_label, self.config.cuda)
        self.test_dataset = self.__construct_dataset(self.test_unified_ids, self.test_label, self.config.cuda)

        if 'inference dataset' in self.config.dataset:
            inference_support_dataset = l2l.data.MetaDataset(self.train_dataset)
            inference_query_dataset = l2l.data.MetaDataset(self.test_dataset)

            self.inference_support_set = l2l.data.TaskDataset(
                dataset=inference_support_dataset,
                task_transforms=[
                    l2l.data.transforms.NWays(inference_support_dataset, n=self.config.inference_way),
                    l2l.data.transforms.KShots(inference_support_dataset, k=self.config.inference_shot),
                    l2l.data.transforms.LoadData(inference_support_dataset),
                ],
                num_tasks=100,
            )

            self.inference_query_set = l2l.data.TaskDataset(
                dataset=inference_query_dataset,
                task_transforms=[
                    l2l.data.transforms.NWays(inference_query_dataset, n=self.config.inference_way),
                    l2l.data.transforms.KShots(inference_query_dataset, k=self.config.inference_query),
                    l2l.data.transforms.LoadData(inference_query_dataset),
                ],
                num_tasks=100,
            )

            if self.config.cuda:
                self.support_samples, self.support_labels, self.query_samples, self.query_labels = torch.cuda.LongTensor(
                    self.train_unified_ids), torch.cuda.LongTensor(self.train_label), torch.cuda.LongTensor(
                    self.test_unified_ids), torch.cuda.LongTensor(self.test_label)
            else:
                self.support_samples, self.support_labels, self.query_samples, self.query_labels = torch.LongTensor(
                    self.train_unified_ids), torch.LongTensor(self.train_label), torch.LongTensor(
                    self.test_unified_ids), torch.LongTensor(self.test_label)

            # for few-shot supervised learning
            support_samples, support_labels = self.inference_support_set.sample()
            query_samples, query_labels = self.inference_query_set.sample()

            train_dataset = self.__construct_dataset(support_samples, support_labels, self.config.cuda)
            test_dataset = self.__construct_dataset(query_samples, query_labels, self.config.cuda)

            self.train_dataloader = self.__construct_dataloader(train_dataset, self.config.batch_size)
            self.test_dataloader = self.__construct_dataloader(test_dataset, self.config.batch_size)
            return None

        '''construct_dataloader'''
        self.train_dataloader = self.__construct_dataloader(self.train_dataset, self.config.batch_size)
        self.test_dataloader = self.__construct_dataloader(self.test_dataset, self.config.batch_size)

    def __process_multiple_classification_dataset(self):
        train_seqs = []
        train_labels = []
        test_seqs = []
        test_labels = []
        for i, peptide in enumerate(self.multiple_classification_dataset):
            # raw_peptide_dataset: [peptide_name, seq_list, label_list]
            self.label_dict[peptide[2]] = peptide[0]
            class_i_seq_list = peptide[1]

            class_i_data_ids, class_i_max_len = self.__token2index(self.token2index, class_i_seq_list)
            class_i_unified_ids = self.__unify_length(class_i_data_ids, self.token2index, self.config.max_len)

            random.seed(self.config.seed)
            random.shuffle(class_i_unified_ids)

            train_num = int(0.8 * len(class_i_unified_ids))
            test_num = len(class_i_unified_ids) - int(0.8 * len(class_i_unified_ids))
            class_i_train_seqs = class_i_unified_ids[:train_num]
            class_i_test_seqs = class_i_unified_ids[train_num:]
            class_i_train_labels = [peptide[2]] * train_num
            class_i_test_labels = [peptide[2]] * test_num

            train_seqs.extend(class_i_train_seqs)
            train_labels.extend(class_i_train_labels)
            test_seqs.extend(class_i_test_seqs)
            test_labels.extend(class_i_test_labels)

        print('#' * 200)
        print(
            'total number of data used for pretrain [{}]: Train[{}] + Test[{}]'.format(len(train_seqs) + len(test_seqs),
                                                                                       len(train_labels),
                                                                                       len(test_labels)))
        print('#' * 200)

        dataset_train = self.__construct_dataset(train_seqs, train_labels, self.config.cuda)
        dataset_test = self.__construct_dataset(test_seqs, test_labels, self.config.cuda)

        self.train_dataloader = self.__construct_dataloader(dataset_train, self.config.batch_size)
        self.test_dataloader = self.__construct_dataloader(dataset_test, self.config.batch_size)

    def __process_cross_validation_dataset(self):
        '''token2index'''
        self.train_data_ids, train_max_len = self.__token2index(self.token2index, self.train_raw_data)
        self.data_max_len = train_max_len

        '''unify_length'''
        self.train_unified_ids = self.__unify_length(self.train_data_ids, self.token2index, self.data_max_len)

        '''Divide Train and Valid Set'''
        self.train_dataloader = []
        self.valid_dataloader = []
        for iter_k in range(self.config.k_fold):
            train_unified_ids = [x for i, x in enumerate(self.train_unified_ids) if
                                 i % self.config.k_fold != iter_k]
            valid_unified_ids = [x for i, x in enumerate(self.train_unified_ids) if
                                 i % self.config.k_fold == iter_k]
            train_label = [x for i, x in enumerate(self.train_label) if i % self.config.k_fold != iter_k]
            valid_label = [x for i, x in enumerate(self.train_label) if i % self.config.k_fold == iter_k]

            '''construct_dataset'''
            self.train_dataset = self.__construct_dataset(train_unified_ids, train_label, self.config.cuda)
            self.valid_dataset = self.__construct_dataset(valid_unified_ids, valid_label, self.config.cuda)
            '''construct_dataloader'''
            self.train_dataloader.append(self.__construct_dataloader(self.train_dataset, self.config.batch_size))
            self.valid_dataloader.append(self.__construct_dataloader(self.valid_dataset, self.config.batch_size))

    def __process_peptide_sequence(self):
        '''construct meta dataset'''
        all_seq_list = []
        all_label_list = []
        for i, peptide in enumerate(self.meta_raw_data):
            # self.meta_raw_data: [peptide_name, seq_list, label_list]
            self.label_dict[i] = peptide[0]
            self.class_label.append(peptide[2])
            class_i_seq_list = peptide[1]
            all_label_list.extend([peptide[2]] * len(class_i_seq_list))

            class_i_data_ids, class_i_max_len = self.__token2index(self.token2index, class_i_seq_list)
            self.data_max_len = max(self.data_max_len, class_i_max_len)
            self.meta_data_ids.append(class_i_data_ids)

        for class_i_data_ids in self.meta_data_ids:
            class_i_unified_ids = self.__unify_length(class_i_data_ids, self.token2index, self.data_max_len)
            self.meta_unified_ids.append(class_i_unified_ids)
            all_seq_list.extend(class_i_unified_ids)

        self.meta_dataset = self.__construct_dataset(all_seq_list, all_label_list)

        self.meta_train_class = [i for i in range(3, 3 + self.config.num_meta_train)] + [0]
        self.meta_valid_class = [i for i in range(3 + self.config.num_meta_train,
                                                  3 + self.config.num_meta_train + self.config.num_meta_valid)] + [1]
        self.meta_test_class = [i for i in range(3 + self.config.num_meta_train + self.config.num_meta_valid,
                                                 3 + self.config.num_meta_train + self.config.num_meta_valid +
                                                 self.config.num_meta_test)] + [2]

        print('*' * 200)
        print('self.meta_train_class', self.meta_train_class)
        print('self.meta_valid_class', self.meta_valid_class)
        print('self.meta_test_class', self.meta_test_class)
        print('*' * 200)
        meta_dataset_train = l2l.data.FilteredMetaDataset(self.meta_dataset, self.meta_train_class)
        meta_dataset_valid = l2l.data.FilteredMetaDataset(self.meta_dataset, self.meta_valid_class)
        meta_dataset_test = l2l.data.FilteredMetaDataset(self.meta_dataset, self.meta_test_class)

        # RemapLabels(FPD_dataset_meta_train, shuffle=True)
        # RemapLabels(FPD_dataset_meta_valid, shuffle=True)
        # RemapLabels(FPD_dataset_meta_test, shuffle=True)

        meta_train_tasks = l2l.data.TaskDataset(
            dataset=meta_dataset_train,
            task_transforms=[
                l2l.data.transforms.NWays(meta_dataset_train, n=self.config.train_way),
                l2l.data.transforms.KShots(meta_dataset_train, k=self.config.train_shot + self.config.train_query),
                l2l.data.transforms.LoadData(meta_dataset_train),
            ],
            num_tasks=10000,
        )
        meta_valid_tasks = l2l.data.TaskDataset(
            dataset=meta_dataset_valid,
            task_transforms=[
                l2l.data.transforms.NWays(meta_dataset_valid, n=self.config.valid_way),
                l2l.data.transforms.KShots(meta_dataset_valid, k=self.config.valid_shot + self.config.valid_query),
                l2l.data.transforms.LoadData(meta_dataset_valid),
            ],
            num_tasks=5000,
        )
        meta_test_tasks = l2l.data.TaskDataset(
            dataset=meta_dataset_test,
            task_transforms=[
                l2l.data.transforms.NWays(meta_dataset_test, n=self.config.test_way),
                l2l.data.transforms.KShots(meta_dataset_test, k=self.config.test_shot + self.config.test_query),
                l2l.data.transforms.LoadData(meta_dataset_test),
            ],
            num_tasks=1000,
        )
        print('len(meta_train_tasks)', len(meta_train_tasks), 'meta_train_tasks', meta_train_tasks)
        print('len(meta_valid_tasks)', len(meta_valid_tasks), 'meta_valid_tasks', meta_valid_tasks)
        print('len(meta_test_tasks)', len(meta_test_tasks), 'meta_test_tasks', meta_test_tasks)

        # meta_train_tasks = None
        # meta_valid_tasks = None
        self.meta_dataloader = [meta_train_tasks, meta_valid_tasks, meta_test_tasks]

    def __process_miniImageNet(self):
        dataset_train = miniimagenet(self.config.path_meta_dataset,
                                     shots=self.config.train_shot,
                                     ways=self.config.train_way,
                                     shuffle=True,
                                     test_shots=self.config.train_query,
                                     meta_train=True,
                                     download=False)
        dataset_valid = miniimagenet(self.config.path_meta_dataset,
                                     shots=self.config.valid_shot,
                                     ways=self.config.valid_way,
                                     shuffle=True,
                                     test_shots=self.config.valid_query,
                                     meta_val=True,
                                     download=False)
        dataset_test = miniimagenet(self.config.path_meta_dataset,
                                    shots=self.config.test_shot,
                                    ways=self.config.test_way,
                                    shuffle=True,
                                    test_shots=self.config.test_query,
                                    meta_test=True,
                                    download=False)

        dataloader_train = BatchMetaDataLoader(dataset_train,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=self.config.num_workers)
        dataloader_valid = BatchMetaDataLoader(dataset_valid,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=self.config.num_workers)
        dataloader_test = BatchMetaDataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=self.config.num_workers)
        self.miniImageNet_dataloader = [dataloader_train, dataloader_valid, dataloader_test]
        print('len(dataloader_train)', len(dataloader_train), 'type(dataloader_train)', type(dataloader_train))
        print('len(dataloader_valid)', len(dataloader_valid), 'type(dataloader_valid)', type(dataloader_valid))
        print('len(dataloader_test)', len(dataloader_test), 'type(dataloader_test)', type(dataloader_test))

        generator_train = self.__batch_task_generator(dataloader_train)
        generator_valid = self.__batch_task_generator(dataloader_valid)
        generator_test = self.__batch_task_generator(dataloader_test)
        self.miniImageNet_generator = [generator_train, generator_valid, generator_test]

    def __batch_task_generator(self, dataloader):
        for i, data in enumerate(dataloader):
            yield data
