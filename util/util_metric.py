import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
import numpy as np


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)  # Default 1 is positive sample
    AUC = auc(FPR, TPR)

    # PRC and AP
    # precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    # AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    # ROC(FPR, TPR, AUC)
    # PRC(Recall, Precision, AP)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC]
    roc_data = None
    prc_data = None
    # roc_data = [FPR, TPR, AUC]
    # prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data


# draw ROC
def ROC(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.ylabel('True Positive Rate', fontdict={'weight': 'normal', 'size': 30})
    plt.title('Receiver operating characteristic github_project', fontdict={'weight': 'normal', 'size': 30})
    plt.legend(loc="lower right", prop={'weight': 'normal', 'size': 30})
    plt.show()


# draw PRC
def PRC(recall, precision, AP):
    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(AP))
    plt.show()
