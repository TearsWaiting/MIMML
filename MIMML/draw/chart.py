# ---encoding:utf-8---
# @Time : 2021.04.12
# @Author : Waiting涙
# @Email : 1773432827@qq.com
# @IDE : PyCharm
# @File : chart.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn import datasets
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


def t_sne_test():
    digits = load_digits()
    X_tsne = TSNE(n_components=2, random_state=500).fit_transform(digits.data)
    print('over')
    X_pca = PCA(n_components=2).fit_transform(digits.data)

    font = {"color": "darkred",
            "size": 13,
            "family": "serif"}

    plt.style.use("dark_background")
    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits.target, alpha=0.6,
    #             cmap=plt.cm.get_cmap('rainbow', 10))
    # plt.title("t-SNE", fontdict=font)
    # cbar = plt.colorbar(ticks=range(10))
    # cbar.set_label(label='digit value', fontdict=font)
    # plt.clim(-0.5, 9.5)
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=digits.target, alpha=0.6,
                cmap=plt.cm.get_cmap('rainbow', 10))
    plt.title("PCA", fontdict=font)
    cbar = plt.colorbar(ticks=range(10))
    cbar.set_label(label='digit value', fontdict=font)
    plt.clim(-0.5, 9.5)
    plt.tight_layout()
    plt.show()


def plot_decision_region(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    points = np.array([xx1.ravel(), xx2.ravel()]).T
    print('points', points.shape, points)
    Z = classifier.predict(points)
    print('Z', Z.shape, Z)
    Z = Z.reshape(xx1.shape)
    print('Z', Z.shape, Z)

    print('xx1', xx1.shape, xx1)
    print('xx2', xx2.shape, xx2)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.show()

    print('X', X.shape, X)
    print('y', y.shape, y)

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


def pairwise_distances_logits(a, b, temperature):
    n = a.shape[0]
    m = b.shape[0]
    logits = -0.5 * temperature * ((a.unsqueeze(1).expand(n, m, -1) -
                                    b.unsqueeze(0).expand(n, m, -1)) ** 2).sum(dim=2)
    return logits

def decision_boundry_test():
    # resolution = 0.05
    # resolution = 0.005
    resolution = 0.001

    X_support = torch.tensor([[0.1, 0.1], [0.2, 0.2],
                              [0.3, 0.5], [0.35, 0.7],
                              [0.7, 0.8], [0.8, 0.7],
                              [0.5, 0.65], [0.56, 0.63],
                              [0.9, 0.9], [0.95, 0.85]])
    X_query = torch.tensor([[0.1, 0.1], [0.2, 0.2],
                            [0.3, 0.5], [0.35, 0.7],
                            [0.7, 0.8], [0.8, 0.7],
                            [0.5, 0.65], [0.56, 0.63],
                            [0.9, 0.9], [0.95, 0.85]]) + 0.1 * torch.rand([10, 2])
    y_support = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y_query = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    # proto_embeddings = torch.tensor([[0.15, 0.15],
    #                                  [0.35, 0.35],
    #                                  [0.55, 0.55],
    #                                  [0.75, 0.75],
    #                                  [0.95, 0.95]])
    proto_embeddings = X_support.reshape(5, 2, -1).mean(dim=1)

    X_support = X_support.cpu().detach().numpy()
    X_query = X_query.cpu().detach().numpy()
    y_support = y_support.cpu().detach().numpy()
    y_query = y_query.cpu().detach().numpy()

    print('X_support', X_support.shape, X_support[:5])
    print('y_support', y_support.shape, y_support)
    print('X_query', X_query.shape, X_query[:5])
    print('y_query', y_query.shape, y_query)
    print('proto_embeddings', proto_embeddings.shape, proto_embeddings)

    # markers = ('s', 'x', 'o', '^', 'v')
    markers = ('o', 'v', 'x')
    # colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    colors = ('blue', 'red', 'lightgreen', 'orange', 'purple')
    print('np.unique(y_support)', np.unique(y_support))
    print('colors[:len(np.unique(y_support))]', colors[:len(np.unique(y_support))])
    mycmap = ListedColormap(colors)
    print('mycmap', mycmap.colors, mycmap.name, mycmap.N, mycmap.monochrome)

    # plot the decision surface
    # x1_min_support, x1_max_support = X_support[:, 0].min() - 0.05, X_support[:, 0].max() + 0.05
    # x2_min_support, x2_max_support = X_support[:, 1].min() - 0.05, X_support[:, 1].max() + 0.05
    # x1_min_query, x1_max_query = X_query[:, 0].min() - 0.05, X_query[:, 0].max() + 0.05
    # x2_min_query, x2_max_query = X_query[:, 1].min() - 0.05, X_query[:, 1].max() + 0.05
    # x1_min = min(x1_min_support, x1_min_query)
    # x1_max = max(x1_max_support, x1_max_query)
    # x2_min = min(x2_min_support, x2_min_query)
    # x2_max = max(x2_max_support, x2_max_query)
    #
    # print('x1_min',x1_min)
    # print('x1_max',x1_max)
    # print('x2_min',x2_min)
    # print('x2_max',x2_max)

    # points_x1, points_x2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                                    np.arange(x2_min, x2_max, resolution))

    points_x1, points_x2 = np.meshgrid(np.arange(-0.1, 1.06, resolution),
                                       np.arange(0, 1, resolution))
    print('points_x1', points_x1.shape, points_x1)
    print('points_x2', points_x2.shape, points_x2)

    points = np.array([points_x1.ravel(), points_x2.ravel()]).T
    print('points np.array', points.shape, points)

    device = torch.device('cpu')
    points = torch.from_numpy(points)
    points = points.to(device)

    print('points tensor', points.shape)

    points_logits = pairwise_distances_logits(points, proto_embeddings, 100)
    pred = points_logits.argmax(dim=1).view(points_logits.size(0))

    print('points_logits', points_logits.shape)
    print('pred', pred.shape)

    zz = pred.cpu().detach().numpy().tolist()
    dict = {}
    for key in zz:
        dict[key] = dict.get(key, 0) + 1
    print('dict', dict)

    Z = pred.reshape(points_x1.shape)
    Z = Z.cpu().detach().numpy()

    print('Z', Z.shape, type(Z), Z)

    # cs = plt.contourf(points_x1, points_x2, Z, levels=[0, 1, 2, 3, 4], alpha=0.3, cmap=cmap)
    # cs = plt.contourf(points_x1, points_x2, Z, levels=[0, 1, 2, 3, 4], alpha=0.3, colors=colors)
    # proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
    # plt.legend(proxy, ["0", "1", "2", "3", "4"])
    # plt.contourf(points_x1, points_x2, Z, alpha=0.3, cmap=cmap)

    # mycmap = plt.cm.brg
    # mycmap = plt.cm.hot
    C = plt.contourf(points_x1, points_x2, Z, 5, alpha=0.3, cmap=mycmap)
    # plt.clabel(C, inline=True, fontsize=12)

    plt.xlim(points_x1.min(), points_x1.max())
    plt.ylim(points_x2.min(), points_x2.max())

    # plot class samples
    for idx, label in enumerate(np.unique(y_support)):
        plt.scatter(x=X_support[y_support == label, 0],
                    y=X_support[y_support == label, 1],
                    s=100,
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[0],
                    label=label,
                    edgecolors='black')

    for idx, label in enumerate(np.unique(y_query)):
        plt.scatter(x=X_query[y_query == label, 0],
                    y=X_query[y_query == label, 1],
                    s=100,
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[1],
                    label=label,
                    edgecolors='black')

    prototype = proto_embeddings.cpu().detach().numpy()
    for idx, p in enumerate(prototype):
        plt.scatter(x=p[0],
                    y=p[1],
                    s=100,
                    alpha=1,
                    c=colors[idx],
                    marker=markers[2],
                    label=idx,
                    edgecolors='black')

    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    # t_sne_test()
    decision_boundry_test()
