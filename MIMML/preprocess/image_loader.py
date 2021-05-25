# ---encoding:utf-8---
# @Time : 2021.03.07
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : image_loader.py

from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

# dataset = omniglot("../data/task_data/", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
# dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

dataset = miniimagenet("../data/task_data/", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
print(dataset)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

# for batch in dataloader:
#     train_inputs, train_targets = batch["train"]
#     print('Train inputs shape: {0}'.format(train_inputs.shape))  # (16, 25, 1, 28, 28)
#     print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)
#
#     test_inputs, test_targets = batch["test"]
#     print('Test inputs shape: {0}'.format(test_inputs.shape))  # (16, 75, 1, 28, 28)
#     print('Test targets shape: {0}'.format(test_targets.shape))  # (16, 75)

print(len(dataloader))
