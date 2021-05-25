# ---encoding:utf-8---
# @Time : 2021.03.17
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : test_torchmeta.py

from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

if __name__ == '__main__':
    dataset = omniglot("../data/task_data/", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
    dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

    for batch in dataloader:
        # print('len(batch)',len(batch)) # 2 对应的就是batch["train"]和batch["test"]

        train_inputs, train_targets = batch["train"]
        print('Train inputs shape: {0}'.format(train_inputs.shape))  # (16, 25, 1, 28, 28)
        print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

        print('train_targets: ',train_targets)

        test_inputs, test_targets = batch["test"]
        print('Test inputs shape: {0}'.format(test_inputs.shape))  # (16, 75, 1, 28, 28)
        print('Test targets shape: {0}'.format(test_targets.shape))  # (16, 75)

        print('test_targets: ',test_targets)

        print('=' * 100)
