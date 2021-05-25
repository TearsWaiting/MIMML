# ---encoding:utf-8---
# @Time : 2021.03.30
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : Transformer_Encoder.py


import torch
import torch.nn as nn
import numpy as np
from configuration import config as configur
import pickle
from util import util_freeze

'''
模型构建
'''


# 目的是构造出一个注意力判断矩阵，一个[batch_size, seq_len, seq_len]的张量
# 其中参与注意力计算的位置被标记为FALSE，将token为[PAD]的位置掩模标记为TRUE
def get_attn_pad_mask(seq):
    # 在BERT中由于是self_attention，seq_q和seq_k内容相同
    # print('-' * 50, '掩模', '-' * 50)

    batch_size, seq_len = seq.size()
    # print('batch_size', batch_size)
    # print('seq_len', seq_len)

    # print('-' * 10, 'test', '-' * 10)
    # print(seq_q.data.shape)
    # print(seq_q.data.eq(0).shape)
    # print(seq_q.data.eq(0).unsqueeze(1).shape)

    # seq_q.data取出张量seq_q的数据
    # seq_q.data.eq(0)是一个和seq_q.data相同shape的张量，seq_q.data对应位置为0时，结果的对应位置为TRUE，否则为FALSE
    # eq(zero) is PAD token 如果等于0，证明那个位置是[PAD]，因此要掩模，计算自注意力时不需要考虑该位置
    # unsqueeze(1)是在维度1处插入一个维度，维度1及其之后的维度往后移，从原来的[batch_size, seq_len]变成[batch_size, 1, seq_len]
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]

    # expand是将某一个shape为1的维度复制为自己想要的数量，这里是从[batch_size, 1, seq_len]将1维度复制seq_len份
    # 结果是变成[batch_size, seq_len, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

    return pad_attn_mask_expand


# 嵌入层
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table)
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding
        # self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

        # 改变初始化权重测试
        # look_up_table = torch.rand([vocab_size, d_model], dtype=torch.float)
        # look_up_table = torch.zeros([vocab_size, d_model], dtype=torch.float)
        # print('old_look_up_table', look_up_table)
        # torch.nn.init.uniform_(look_up_table, a=0, b=1)
        # torch.nn.init.normal_(look_up_table)
        # print('new_look_up_table', look_up_table)
        # self.tok_embed = self.tok_embed.from_pretrained(look_up_table)
        # print('self.tok_embed.weight', self.tok_embed.weight)
        # self.tok_embed.weight.requires_grad_(True)
        # print('self.tok_embed.weight.requires_grad', self.tok_embed.weight.requires_grad)

    def forward(self, x):
        # print('x.device', x.device)
        seq_len = x.size(1)  # x: [batch_size, seq_len]

        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
        # print('pos.device', pos.device)
        # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

        # expand_as类似于expand，只是目标规格是x.shape
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]

        # 混合3种嵌入向量
        # embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        # embedding = self.tok_embed(x) + self.pos_embed(pos)

        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)

        # layerNorm
        embedding = self.norm(embedding)
        return embedding


# 计算Self-Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        #  这里接收的Q, K, V是q_s, q_k, q_v，也就是真正的Q, K, V是向量，shape:[bach_size, seq_len, d_model]
        # Q: [batch_size, n_head, seq_len, d_k]
        # K: [batch_size, n_head, seq_len, d_k]
        # V: [batch_size, n_head, seq_len, d_v]

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]

        # mask_filled:是将mask中为1/TRUE的元素所在的索引，在原tensor中相同的的索引处替换为指定的value
        # remark: mask必须是一个 ByteTensor而且shape必须和a一样，mask value必须同为tensor
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.

        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        #  这里接收的Q, K, V都是enc_inputs，也就是embedding后的输入，shape:[bach_size, seq_len, d_model]
        # Q: [batch_size, seq_len, d_model]
        # K: [batch_size, seq_len, d_model]
        # V: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)

        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # 多头注意力是同时计算的，一次tensor乘法即可，这里是将多头注意力进行切分

        # print('Q', Q.size(), Q)
        # print('self.W_Q(Q)', self.W_Q(Q))
        # print('self.W_Q(Q).view(batch_size, -1, n_head, d_k)', self.W_Q(Q).view(batch_size, -1, n_head, d_k))
        # print('self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)',
        #       self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2))

        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]

        # 处理前attn_mask: [batch_size, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        # 处理后attn_mask: [batch_size, n_head, seq_len, seq_len]

        # context: [batch_size, n_head, seq_len, d_v], attn: [batch_size, n_head, seq_len, seq_len]
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]

        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


# 基于位置的全连接层
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 多头注意力模块
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        # 全连接模块
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


# 完整的模型
class Transformer_Encoder(nn.Module):
    def __init__(self, config):
        super(Transformer_Encoder, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device
        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d_model = config.dim_embedding
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = config.vocab_size
        device = torch.device("cuda" if config.cuda else "cpu")
        ways = config.train_way
        # ways = config.ways

        print('BERT definition: max_len', max_len)

        # Embedding Layer
        self.embedding = Embedding()

        # Encoder Layer
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # 定义重复的模块

        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            # nn.Linear(d_model, 2),
        )

        # self.classifier = nn.Linear(d_model, ways)

    def forward(self, input_ids):
        # embedding layer
        output = self.embedding(input_ids)  # [bach_size, seq_len, d_model]
        # print('output', output.size())

        # 获取掩模判断矩阵
        enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]

        # encoder layer
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)
            # output: [batch_size, max_len, d_model]

        # task-specific layer
        # 只取[CLS]
        output = output[:, 0, :]
        embeddings = self.fc_task(output)
        embeddings = embeddings.view(embeddings.size(0), -1)
        logits_clsf = None
        # logits_clsf = self.classifier(embeddings)

        return logits_clsf, embeddings


def check_model():
    config = configur.get_train_config()
    torch.cuda.set_device(config.device)  # 选择要使用的GPU

    # 加载词典
    residue2idx = pickle.load(open(config.path_meta_data + 'residue2idx.pkl', 'rb'))
    config.vocab_size = len(residue2idx)

    model = Transformer_Encoder(config)

    print('-' * 50, 'Model', '-' * 50)
    print(model)

    print('-' * 50, 'Model.named_parameters', '-' * 50)
    for name, value in model.named_parameters():
        print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))

    # 冻结网络的部分层
    # util_freeze.freeze_by_names(model, ['embedding', 'layers'])
    util_freeze.freeze_by_idxs(model, [0, 1])

    print('-' * 50, 'Model.named_children', '-' * 50)
    for name, child in model.named_children():
        print('\\' * 40, '[name:{}]'.format(name), '\\' * 40)
        print('child:\n{}'.format(child))

        if name == 'soft_attention':
            print('soft_attention')
            for param in child.parameters():
                print('param.shape', param.shape)
                print('param.requires_grad', param.requires_grad)

        for sub_name, sub_child in child.named_children():
            print('*' * 20, '[sub_name:{}]'.format(sub_name), '*' * 20)
            print('sub_child:\n{}'.format(sub_child))

            # if name == 'layers' and (sub_name == '5' or sub_name == '4'):
            if name == 'layers' and (sub_name == '5'):
                print('Ecoder 5 is unfrozen')
                for param in sub_child.parameters():
                    param.requires_grad = True

        # for param in child.parameters():
        #     print('param.requires_grad', param.requires_grad)

    print('-' * 50, 'Model.named_parameters', '-' * 50)
    for name, value in model.named_parameters():
        print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))


def forward_test():
    config = configur.get_train_config()
    torch.cuda.set_device(config.device)  # 选择要使用的GPU

    # 加载词典
    residue2idx = pickle.load(open(config.path_meta_data + 'residue2idx.pkl', 'rb'))
    config.vocab_size = len(residue2idx)

    model = Transformer_Encoder(config)

    input = torch.randint(28, [4, 20])

    if config.cuda:
        device = torch.device('cuda')
        model = model.to(device)
        input = input.to(device)

    output = model(input)
    print('output', output)


if __name__ == '__main__':
    # check model
    check_model()
    # forward test
    forward_test()
