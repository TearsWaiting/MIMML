# ---encoding:utf-8---
# @Time : 2021.01.14
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : TextRCNN.py


import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return torch.max_pool1d(x, kernel_size=x.shape[-1])


class TextRCNN(nn.Module):
    def __init__(self, config):
        super(TextRCNN, self).__init__()
        vocab_size = config.vocab_size
        embedding_dim = config.dim_embedding
        hidden_size = config.hidden_size
        num_labels = config.num_class

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.globalmaxpool = GlobalMaxPool1d()
        # self.dropout = nn.Dropout(.5)
        self.dropout = nn.Dropout(.3)
        self.linear1 = nn.Linear(embedding_dim + 2 * hidden_size, 256)
        self.linear2 = nn.Linear(256, num_labels)

    def forward(self, x):  # x: [batch,L]
        x_embed = self.embed(x)  # x_embed: [batch,L,embedding_size]
        last_hidden_state, (c, h) = self.lstm(x_embed)  # last_hidden_state: [batch,L,hidden_size * num_bidirectional]
        out = torch.cat((x_embed, last_hidden_state),
                        2)  # out: [batch,L,embedding_size + hidden_size * num_bidirectional]
        # print(out.shape)
        out = F.relu(self.linear1(out))
        out = out.permute(dims=[0, 2, 1])  # out: [batch,embedding_size + hidden_size * num_bidirectional,L]
        out = self.globalmaxpool(out).squeeze(-1)  # out: [batch,embedding_size + hidden_size * num_bidirectional]
        # print(out.shape)
        out = self.dropout(out)  # out: [batch,embedding_size + hidden_size * num_bidirectional]
        out = self.linear2(out)  # out: [batch,num_labels]
        return out, None
