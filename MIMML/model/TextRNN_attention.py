import torch
import torch.nn as nn
from model.Attention import Attention


# soft_attention机制
class soft_attention(nn.Module):
    def __init__(self, config):
        super(soft_attention, self).__init__()
        self.seq_len = 41
        self.hidden_size = config.dim_embedding
        self.embed_atten_size = config.embed_atten_size

        self.atten = nn.Linear(self.hidden_size, self.embed_atten_size)
        self.merge = nn.Linear(self.embed_atten_size, 1)

    def forward(self, embedding_vector):
        # print('[{}.shape]-->{}'.format('embedding_input', embedding_vector.shape))
        # embedding_vector: [vocab_size, num_embedding, d_model]

        input_reshape = torch.Tensor.reshape(embedding_vector, [-1, self.hidden_size])
        # output_reshape: [batch_size * sequence_length, hidden_size]
        # print('[{}.shape]-->{}'.format('input_reshape', input_reshape.shape))

        attn_tanh = self.atten(input_reshape)
        # attn_tanh = torch.tanh(torch.mm(input_reshape, self.w_omega))
        # attn_tanh: [batch_size * sequence_length, embed_atten_size]
        # print('[{}.shape]-->{}'.format('attn_tanh', attn_tanh.shape))

        attn_hidden_layer = self.merge(attn_tanh)
        # attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # attn_hidden_layer: [batch_size * sequence_length, 1]
        # print('[{}.shape]-->{}'.format('attn_hidden_layer', attn_hidden_layer.shape))

        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.seq_len])
        # exps: [batch_size, sequence_length]
        # print('[{}.shape]-->{}'.format('exps', exps.shape))

        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # alphas: [batch_size, sequence_length]
        # print('[{}.shape]-->{}'.format('alphas', alphas.shape))

        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.seq_len, 1])
        # alphas_reshape: [batch_size, sequence_length, 1]
        # print('[{}.shape]-->{}'.format('alphas_reshape', alphas_reshape.shape))

        attn_output = torch.sum(embedding_vector * alphas_reshape, 1)
        # attn_output: [batch_size, hidden_size]
        # print('[{}.shape]-->{}'.format('attn_output', attn_output.shape))

        return attn_output


# 循环神经网络 (many-to-one)
class TextRNN_Attention(nn.Module):
    def __init__(self, config):
        super(TextRNN_Attention, self).__init__()
        embedding_dim = config.dim_embedding
        label_num = config.num_class
        vocab_size = config.vocab_size
        batch_size = config.batch_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_num = config.num_layer
        self.bidirectional = config.bidirectional
        self.device = torch.device("cuda" if config.cuda else "cpu")

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if config.static:  # 如果使用预训练词向量，则提前加载，当不需要微调时设置freeze为True
            self.embedding = self.embedding.from_pretrained(config.vectors, freeze=not config.fine_tune)

        self.lstm = nn.LSTM(embedding_dim,  # x的特征维度,即embedding_dim
                            self.hidden_size,  # 隐藏层单元数
                            self.layer_num,  # 层数
                            batch_first=True,  # 将输入第一个维度设为 batch, 即:(batch_size, seq_length, embedding_dim)
                            bidirectional=self.bidirectional,  # 是否用双向
                            dropout=config.dropout  # dropout概率，默认为0
                            )
        self.attn = Attention(self.hidden_size * 2 if self.bidirectional else self.hidden_size)
        self.soft_attention = soft_attention(config)
        self.fc = nn.Linear(self.hidden_size * 2, label_num) if self.bidirectional else nn.Linear(self.hidden_size,
                                                                                                  label_num)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        # 输入x的维度为(batch_size, max_len), max_len可以通过torchtext设置或自动获取为训练样本的最大长度
        x = self.embedding(x)  # 经过embedding,x的维度为(batch_size, time_step, input_size=embedding_dim)

        # 隐层初始化
        # h0维度为(num_layers*direction_num, batch_size, hidden_size)
        # c0维度为(num_layers*direction_num, batch_size, hidden_size)
        # h0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size,
        #                  device=self.device) if self.bidirectional else torch.zeros(
        #     self.layer_num, x.size(0), self.hidden_size, device=self.device)
        #
        # c0 = torch.zeros(self.layer_num * 2, x.size(0), self.hidden_size,
        #                  device=self.device) if self.bidirectional else torch.zeros(
        #     self.layer_num, x.size(0), self.hidden_size, device=self.device)

        # LSTM前向传播，此时out维度为(batch_size, seq_length, hidden_size*direction_num)
        # hn,cn表示最后一个状态?维度与h0和c0一样
        # out, (hn, cn) = self.lstm(x, (h0, c0))

        out, (hn, cn) = self.lstm(x)

        # 直接预测
        # out = self.fc(out[:, -1, :])

        # 输出维度分析，记得batch_size和seq_len倒置了
        # out = out.view([self.config.batch_size, -1, self.config.dim_embedding])
        # print('out.shape', out.shape)
        # print('hn.shape', hn.shape)
        # print('cn.shape', cn.shape)

        # soft_attention机制
        attn_out = self.attn(out, torch.tensor([x.size(1)]))
        # attn_out = self.soft_attention(out)
        # print('attn_out.shape', attn_out.shape)
        hidden = self.dropout(attn_out)
        out = self.fc(hidden)

        return out, None
