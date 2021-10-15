import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from model import Transformer_Encoder, Convnet, TextCNN
from util import util_metric
from matplotlib.colors import ListedColormap


class ProtoNet(nn.Module):
    def __init__(self, config):
        super(ProtoNet, self).__init__()
        self.config = config
        self.backbone = None
        if self.config.backbone == 'Transformer Encoder':
            self.backbone = Transformer_Encoder.Transformer_Encoder(self.config)
        elif self.config.backbone == 'Convnet':
            self.backbone = Convnet.Convnet()
        elif self.config.backbone == 'TextCNN':
            self.backbone = TextCNN.TextCNN(self.config)
        else:
            print('Error, No Such Meta Model Backbone')
        self.if_MIM = self.config.if_MIM

    def process_data(self, task, way, shot, query):
        data, label = task
        # print('data', data.size())
        # print('label', label.size(), label)
        # data = [x.view(-1, 1) for x in data]
        # data = torch.cat(data, dim=1)

        reset_label = np.repeat(np.arange(way), shot + query, 0)
        reset_label = torch.from_numpy(reset_label)
        if self.config.cuda:
            # reset_label = reset_label.to(device)
            reset_label = reset_label.cuda()

        # Sort data samples by labels
        sort = torch.sort(reset_label)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        reset_label = reset_label.squeeze(0)[sort.indices].squeeze(0)

        return data, reset_label

    def pairwise_distances_logits(self, a, b, temperature):
        n = a.shape[0]
        m = b.shape[0]
        logits = -0.5 * temperature * ((a.unsqueeze(1).expand(n, m, -1) -
                                        b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
        return logits

    def get_accuracy(self, predictions, targets):
        predictions = predictions.argmax(dim=1).view(targets.shape)
        return (predictions == targets).sum().float() / targets.size(0)

    def get_MI(self, probs):
        cond_ent = self.get_cond_entropy(probs)
        ent = self.get_entropy(probs)
        return ent - cond_ent

    def get_entropy(self, probs):
        ent = - (probs.mean(0) * torch.log2(probs.mean(0) + 1e-12)).sum(0, keepdim=True)
        return ent

    def get_cond_entropy(self, probs):
        cond_ent = - (probs * torch.log(probs + 1e-12)).sum(1).mean(0, keepdim=True)
        return cond_ent

    def draw_decision_boundary(self, X, y, proto_embeddings, resolution=0.002):
        X_support, X_query = X
        y_support, y_query = y

        X_support = X_support.cpu().detach().numpy()
        X_query = X_query.cpu().detach().numpy()
        y_support = y_support.cpu().detach().numpy()
        y_query = y_query.cpu().detach().numpy()

        plt.figure(11, figsize=(8, 6))
        # plt.figure(11, figsize=(8, 6), dpi=300)
        # markers = ('s', 'x', 'o', '^', 'v')
        markers = ('o', '^', '*')
        colors = ('#99DEF1', '#F1A497', '#BBF09A', '#F9E1C9', '#DACDF5')
        cmap = ListedColormap(colors[:len(np.unique(y_support))])

        # plot the decision surface
        x1_min_support, x1_max_support = X_support[:, 0].min() - 0.1 - 0.3, X_support[:, 0].max() + 0.1
        x2_min_support, x2_max_support = X_support[:, 1].min() - 0.1, X_support[:, 1].max() + 0.1
        x1_min_query, x1_max_query = X_query[:, 0].min() - 0.1 - 0.3, X_query[:, 0].max() + 0.1
        x2_min_query, x2_max_query = X_query[:, 1].min() - 0.1, X_query[:, 1].max() + 0.1
        x1_min = min(x1_min_support, x1_min_query)
        x1_max = max(x1_max_support, x1_max_query)
        x2_min = min(x2_min_support, x2_min_query)
        x2_max = max(x2_max_support, x2_max_query)

        points_x1, points_x2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                           np.arange(x2_min, x2_max, resolution))

        points = np.array([points_x1.ravel(), points_x2.ravel()]).T
        device = torch.device('cuda') if self.config.cuda else torch.device('cpu')
        points = torch.from_numpy(points).to(device)
        points_logits = self.pairwise_distances_logits(points, proto_embeddings, self.config.temp)
        pred = points_logits.argmax(dim=1).view(points_logits.size(0))

        Z = pred.reshape(points_x1.shape)
        Z = Z.cpu().detach().numpy()
        plt.contourf(points_x1, points_x2, Z, 5, alpha=0.3, cmap=cmap)

        plt.xlim(points_x1.min(), points_x1.max())
        plt.ylim(points_x2.min(), points_x2.max())

        # plot class samples
        for idx, label in enumerate(np.unique(y_support)):
            plt.scatter(x=X_support[y_support == label, 0],
                        y=X_support[y_support == label, 1],
                        s=100,
                        alpha=0.6,
                        c=colors[idx],
                        marker=markers[0],
                        label=label,
                        edgecolors='black')

        for idx, label in enumerate(np.unique(y_query)):
            plt.scatter(x=X_query[y_query == label, 0],
                        y=X_query[y_query == label, 1],
                        s=80,
                        alpha=0.6,
                        c=colors[idx],
                        marker=markers[1],
                        label=label,
                        edgecolors='black')

        prototype = proto_embeddings.cpu().detach().numpy()
        for idx, p in enumerate(prototype):
            plt.scatter(x=p[0],
                        y=p[1],
                        s=180,
                        alpha=0.9,
                        c=colors[idx],
                        marker=markers[2],
                        label=idx,
                        edgecolors='black')

        plt.xticks(fontproperties='Times New Roman', size=13)
        plt.yticks(fontproperties='Times New Roman', size=13)
        # plt.xlabel('Dimension 1', fontsize=15)
        # plt.ylabel('Dimension 2', fontsize=15)
        plt.legend(loc='upper left')
        font = {"color": "darkred", "size": 18, "family": "serif"}
        plt.title("{}".format(self.config.title), fontdict=font)
        plt.savefig(self.config.path_save + self.config.learn_name + '/{}.pdf'.format(self.config.title))
        plt.show()

    def fast_adapt(self, task, way, shot, query, if_transductive, if_meta_train, visual=False):
        if self.config.dataset == 'Peptide Sequence':
            data, label = self.process_data(task, way, shot, query)

            # Compute support and query embeddings
            if self.config.backbone == 'Transformer Encoder':
                embeddings = self.backbone(data)[1]
            elif self.config.backbone == 'TextCNN':
                embeddings = self.backbone(data)
            else:
                embeddings = None
                print('Mo Such Model')

            # Normalize embeddings
            embeddings = F.normalize(embeddings, dim=1)

            # Compute indices
            support_indices = np.zeros(data.size(0), dtype=bool)
            selection = np.arange(way) * (shot + query)
            for offset in range(shot):
                support_indices[selection + offset] = True
            query_indices = torch.from_numpy(~support_indices)
            support_indices = torch.from_numpy(support_indices)

            # Compute embeddings according to indices
            support_embeddings = embeddings[support_indices]
            # support_embeddings: [shot, dim_feature]
            proto_embeddings = support_embeddings.reshape(way, shot, -1).mean(dim=1)
            # proto_embeddings: [way, dim_feature]
            query_embeddings = embeddings[query_indices]
            # support: [query_num, dim_feature]
            support_labels = label[support_indices].long()
            # query_labels: [shot]
            query_labels = label[query_indices].long()
            # query_labels: [query_num]
        elif self.config.dataset == 'miniImageNet' or 'inference dataset' in self.config.dataset:
            support_samples, support_labels, query_samples, query_labels = task

            support_embeddings = self.backbone(support_samples)
            query_embeddings = self.backbone(query_samples)

            support_labels = support_labels.long()
            support_sort = torch.sort(support_labels)
            support_embeddings = torch.index_select(support_embeddings, 0, support_sort.indices)
            support_labels = torch.index_select(support_labels, 0, support_sort.indices)

            if self.config.dataset == 'imbalanced inference dataset':
                num_pos = torch.sum(support_labels)
                num_neg = support_labels.size(0) - num_pos
                support_pos = support_embeddings[num_neg:]
                support_neg = support_embeddings[:num_neg]
                proto_pos = support_pos.mean(dim=0).unsqueeze(0)
                proto_neg = support_neg.mean(dim=0).unsqueeze(0)
                proto_embeddings = torch.cat([proto_neg, proto_pos], dim=0)
            else:
                proto_embeddings = support_embeddings.reshape(way, shot, -1).mean(dim=1)

            query_labels = query_labels.long()
            query_sort = torch.sort(query_labels)
            query_embeddings = torch.index_select(query_embeddings, 0, query_sort.indices)
            query_labels = torch.index_select(query_labels, 0, query_sort.indices)
        else:
            print('Error, No Such Dataset')

        # Compute logits, CE loss, acc
        support_logits = self.pairwise_distances_logits(support_embeddings, proto_embeddings,
                                                        self.config.temp)  # support_logits: [shot, way]
        loss_support_CE = (F.cross_entropy(support_logits, support_labels).float()).mean()
        support_acc = self.get_accuracy(support_logits, support_labels)

        query_logits = self.pairwise_distances_logits(query_embeddings, proto_embeddings,
                                                      self.config.temp)  # query_logits: [query_num, way]
        loss_query_CE = (F.cross_entropy(query_logits, query_labels).float()).mean()
        query_acc = self.get_accuracy(query_logits, query_labels)

        if not self.if_MIM:
            support_mi = torch.tensor(0)
            query_mi = torch.tensor(0)
            if if_meta_train:
                loss_sum = loss_query_CE
            else:
                loss_sum = loss_support_CE
        else:
            # Compute softmax
            support_probs = support_logits.softmax(1)
            query_probs = query_logits.softmax(1)

            # Compute entropy, conditional entropy and mutual information
            support_ent = self.get_entropy(probs=support_probs)
            support_cond_ent = self.get_cond_entropy(probs=support_probs)
            support_mi = support_ent - support_cond_ent

            query_ent = self.get_entropy(probs=query_probs)
            query_cond_ent = self.get_cond_entropy(probs=query_probs)
            query_mi = query_ent - query_cond_ent

            if if_transductive:
                ''' transductive training & transductive inference '''
                if if_meta_train:
                    '''L_S_CE + L_Q_CE + L_S_MI + L_Q_MI'''
                    loss_sum = self.config.lamb * (loss_support_CE + loss_query_CE) - \
                               (support_ent - self.config.alpha * support_cond_ent) - \
                               (query_ent - self.config.alpha * query_cond_ent)

                    # meta-train
                    # loss_sum: tranductive CE and MI loss in query set
                    # support_mi: mutual information in support set
                    # query_mi: mutual information in query set
                    # support_acc: accuracy in support set
                    # query_acc: accuracy in query set
                    # return loss_sum, support_mi, query_mi, support_acc, query_acc
                else:
                    '''L_S_CE + L_S_MI + L_Q_MI'''
                    loss_sum = self.config.lamb * loss_support_CE - \
                               (support_ent - self.config.alpha * support_cond_ent) - \
                               (query_ent - self.config.alpha * query_cond_ent)

                    '''L_S_CE + L_Q_MI'''
                    # loss_sum = self.config.lamb * loss_support_CE - \
                    #            (query_ent - self.config.alpha * query_cond_ent)

                    '''L_S_CE + L_S_MI'''
                    # loss_sum = self.config.lamb * loss_support_CE - \
                    #            (support_ent - self.config.alpha * support_cond_ent)

                    '''L_S_MI + L_Q_MI'''
                    # loss_sum = - (support_ent - self.config.alpha * support_cond_ent) - \
                    #            (query_ent - self.config.alpha * query_cond_ent)

                    '''L_S_CE'''
                    # loss_sum = self.config.lamb * loss_support_CE

                    '''L_S_MI'''
                    # loss_sum = - (support_ent - self.config.alpha * support_cond_ent)

                    '''L_Q_MI'''
                    # loss_sum = - (query_ent - self.config.alpha * query_cond_ent)

                    # meta-test
                    # loss_sum: inductive CE loss in support set + tranductive MI loss in query set
                    # support_mi: mutual information in support set
                    # query_mi: mutual information in query set
                    # support_acc: accuracy in support set
                    # query_acc: accuracy in query set
                    # return loss_sum, support_mi, query_mi, support_acc, query_acc
            else:
                ''' transductive training & inductive inference '''
                if if_meta_train:
                    '''L_S_CE + L_Q_CE + L_S_MI'''
                    loss_sum = self.config.lamb * (loss_support_CE + loss_query_CE) - \
                               (support_ent - self.config.alpha * support_cond_ent)
                else:
                    '''L_S_CE + L_S_MI'''
                    loss_sum = self.config.lamb * loss_support_CE - \
                               (support_ent - self.config.alpha * support_cond_ent)

        if 'inference dataset' in self.config.dataset:
            query_pred_label = query_logits.argmax(dim=1).view(query_labels.shape).detach().cpu()
            query_labels = query_labels.detach().cpu()
            query_probs = query_probs[:, 1]
            query_probs = query_probs.detach().cpu()
            metric, roc_data, prc_data = util_metric.caculate_metric(query_probs, query_pred_label, query_labels)
            return loss_sum, loss_query_CE, support_mi, query_mi, support_acc, query_acc, metric, query_probs

        # visualization
        # if visual:
        #     draw_embeddings = [support_embeddings, query_embeddings]
        #     draw_labels = [support_labels, query_labels]
        #     self.draw_decision_boundary(draw_embeddings, draw_labels, proto_embeddings)

        return loss_sum, loss_query_CE, support_mi, query_mi, support_acc, query_acc
