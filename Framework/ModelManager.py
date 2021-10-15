import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from model import Transformer_Encoder, Focal_Loss, ProtoNet, TextCNN
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from copy import deepcopy
from util import util_model


class ModelManager():
    def __init__(self, learner):
        self.learner = learner
        self.IOManager = learner.IOManager
        self.visualizer = learner.visualizer
        self.dataManager = learner.dataManager
        self.config = learner.config

        self.mode = self.config.mode
        self.model = None
        self.optimizer = None
        self.loss_func = None

        # supervised learning
        self.best_performance = None
        self.test_performance = []
        self.valid_performance = []
        self.avg_test_loss = 0

        # meta learning
        self.has_save_episodic_state_dict = False
        self.best_meta_test_performance = None
        self.meta_test_performance = []
        self.if_visual = False

    def init_model(self):
        if self.mode == 'train-test' or self.mode == 'cross validation':
            if self.config.model == 'Transformer Encoder':
                self.model = Transformer_Encoder.Transformer_Encoder(self.config)
            elif self.config.model == 'TextCNN':
                self.model = TextCNN.TextCNN(self.config)
            else:
                self.IOManager.log.Error('No Such Model')
            if self.config.cuda:
                self.model.cuda()
        elif self.mode == 'meta learning':
            if self.config.model == 'ProtoNet':
                self.model = ProtoNet.ProtoNet(self.config)
            else:
                self.IOManager.log.Error('No Such Model')
            if self.config.cuda:
                # self.model.to(self.dataManager.device)
                self.model.cuda()
        else:
            self.IOManager.log.Error('No Such Mode')

    def load_params(self):
        if self.config.path_params:
            if self.mode == 'train-test' or self.mode == 'cross validation':
                self.model = self.__load_params(self.model, self.config.path_params)
            elif self.mode == 'meta learning':
                self.model.backbone = self.__load_params(self.model.backbone, self.config.path_params)
                print('meta model load over')
            else:
                self.IOManager.log.Error('No Such Mode')
        else:
            self.IOManager.log.Warn('Path of Parameters Not Exist')

    def adjust_model(self):
        self.model = self.__adjust_model(self.model)

    def init_optimizer(self):
        if self.config.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config.lr,
                                               weight_decay=self.config.reg)
        elif self.config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.lr,
                                              weight_decay=self.config.reg)
        else:
            self.IOManager.log.Error('No Such Optimizer')

    def def_loss_func(self):
        if self.mode == 'train-test' or self.mode == 'cross validation':
            if self.config.loss_func == 'CE':
                self.loss_func = nn.CrossEntropyLoss()
            elif self.config.loss_func == 'FL':
                if self.config.alpha != None:
                    alpha = torch.tensor([self.config.alpha, 1 - self.config.alpha])
                    if self.config.cuda:
                        alpha.cuda()
                else:
                    alpha = None
                self.loss_func = Focal_Loss.FocalLoss(self.config.num_class, alpha=alpha, gamma=self.config.gamma)
            else:
                self.IOManager.log.Error('No Such Loss Function')
        elif self.mode == 'meta learning':
            pass
        else:
            self.IOManager.log.Error('No Such Mode')

    def train(self):
        self.model.train()
        if self.mode == 'train-test':
            train_dataloader = self.dataManager.get_dataloder(name='train_set')
            test_dataloader = self.dataManager.get_dataloder(name='test_set')
            best_performance = self.__SL_train(train_dataloader, test_dataloader)
            self.best_performance = best_performance
            self.IOManager.log.Info('Best Performance: {}'.format(self.best_performance))
            self.IOManager.log.Info('Performance: {}'.format(self.test_performance))
        elif self.mode == 'cross validation':
            for k in range(self.config.k_fold):
                train_dataloader = self.dataManager.get_dataloder(name='train_set')[k]
                valid_dataloader = self.dataManager.get_dataloder(name='valid_set')[k]
                best_valid_performance = self.__cross_validation(train_dataloader, valid_dataloader)
                self.valid_performance.append(best_valid_performance)
            self.IOManager.log.Info('valid_performance: ', self.valid_performance)
        elif self.mode == 'meta learning':
            meta_train_tasks = self.dataManager.get_dataloder(name='meta_train')
            meta_valid_tasks = self.dataManager.get_dataloder(name='meta_valid')
            best_meta_test_performance = self.__meta_train(meta_train_tasks, meta_valid_tasks)
            self.best_meta_test_performance = best_meta_test_performance
            self.IOManager.log.Info('Best Meta-Test Performance: {}'.format(self.best_meta_test_performance))
            self.IOManager.log.Info('Meta-Test Performance: {}'.format(self.meta_test_performance))
        else:
            self.IOManager.log.Error('No Such Mode')

    def test(self):
        self.model.eval()
        if self.mode == 'train-test' or self.mode == 'cross validation':
            test_dataloader = self.dataManager.get_dataloder('test_set')
            if test_dataloader is not None:
                test_performance, avg_test_loss = self.__SL_test(test_dataloader)
                log_text = '\n' + '=' * 20 + ' Final Test Performance ' + '=' * 20 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                    test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                    test_performance[4]) \
                           + '\n' + '=' * 60
                self.IOManager.log.Info(log_text)
            else:
                self.IOManager.log.Warn('Test Data is None.')
        elif self.mode == 'meta learning':
            meta_test_tasks = self.dataManager.get_dataloder(name='meta_test')
            meta_test_performance = self.__meta_test(meta_test_tasks, self.config.test_iteration, self.config.test_way,
                                                     self.config.test_shot, self.config.test_query)
            log_text = '\n' + '=' * 50 + ' Final Meta-Test Performance ' + '=' * 50 \
                       + '\navg_loss_sum={:.3f} | avg_loss_query_CE={:.3f} | query_acc={:.2f}% | test_mi={:.3f} | support_acc={:.2f}% | support_mi={:.3f}'.format(
                meta_test_performance[0], meta_test_performance[1], 100 * meta_test_performance[2],
                meta_test_performance[3], 100 * meta_test_performance[4], meta_test_performance[5]) \
                       + '\n' + '=' * 120
            self.IOManager.log.Info(log_text)
        else:
            self.IOManager.log.Error('No Such Mode')

    def inference(self, type):
        if type == 'Meta Inference':
            self.__meta_inference()
        elif type == 'Few-shot SL':
            self.__few_shot_SL()

    def __load_params(self, model, param_path):
        pretrained_dict = torch.load(param_path, map_location={'cuda:1': 'cuda:0'})
        # print('pretrained_dict', pretrained_dict)
        new_model_dict = model.state_dict()
        # print('new_model_dict', new_model_dict)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
        new_model_dict.update(pretrained_dict)
        # print('new_model_dict after load', new_model_dict)
        model.load_state_dict(new_model_dict)
        return model

    def __adjust_model(self, model, freeze=None):
        print('-' * 50, 'Model.named_parameters', '-' * 50)
        for name, value in model.named_parameters():
            print('[{}]->[{}],[requires_grad:{}]'.format(name, value.shape, value.requires_grad))

        # Count the total parameters
        params = list(model.parameters())
        k = 0
        for i in params:
            l = 1
            for j in i.size():
                l *= j
            k = k + l
        print('=' * 50, "Number of total parameters:" + str(k), '=' * 50)

        if freeze:
            for name in freeze:
                util_model.unfreeze_by_names(model, name)
        return model

    def __get_loss(self, logits, label):
        loss = 0
        if self.config.loss_func == 'CE':
            loss = self.loss_func(logits.view(-1, self.config.num_class), label.view(-1))
            loss = (loss.float()).mean()
            loss = (loss - self.config.b).abs() + self.config.b  # flooding method
        elif self.config.loss_func == 'FL':
            # logits = logits.view(-1, 2)
            # label = label.view(-1)
            loss = self.loss_func(logits, label)
        return loss

    def __caculate_metric(self, pred_prob, label_pred, label_real):
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

        # print('tp\tfp\ttn\tfn')
        # print('{}\t{}\t{}\t{}'.format(tp, fp, tn, fn))

        # Accuracy
        ACC = float(tp + tn) / test_num

        # Precision
        # if tp + fp == 0:
        #     Precision = 0
        # else:
        #     Precision = float(tp) / (tp + fp)

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

        # F1-score
        # if Recall + Precision == 0:
        #     F1 = 0
        # else:
        #     F1 = 2 * Recall * Precision / (Recall + Precision)

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

    def __cross_validation(self, train_dataloader, valid_dataloader):
        step = 0
        best_acc = 0
        best_mcc = 0
        best_valid_performance = 0

        for epoch in range(1, self.config.epoch + 1):
            for batch in train_dataloader:
                data, label = batch
                logits, output = self.model(data)
                train_loss = self.__get_loss(logits, label)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                step += 1

                '''Periodic Train Log'''
                if step % self.config.interval_log == 0:
                    corrects = (torch.max(logits, 1)[1] == label).sum()
                    the_batch_size = label.shape[0]
                    train_acc = 100.0 * corrects / the_batch_size
                    print('Epoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, step, train_loss,
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            the_batch_size))
                    self.visualizer.step_log_interval.append(step)
                    self.visualizer.train_metric_record.append(train_acc)
                    self.visualizer.train_loss_record.append(train_loss)

            '''Periodic Valid'''
            if epoch % self.config.interval_valid == 0:
                valid_performance, avg_test_loss = self.__SL_test(valid_dataloader)
                self.visualizer.step_test_interval.append(epoch)
                self.visualizer.test_metric_record.append(valid_performance[0])
                self.visualizer.test_loss_record.append(avg_test_loss)

                # valid_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
                if self.config.metric == 'ACC':
                    valid_acc = valid_performance[0]
                    if valid_acc > best_acc:
                        best_acc = valid_acc
                        best_valid_performance = valid_performance
                    if self.config.save_best and valid_acc > self.config.threshold:
                        self.IOManager.save_model_dict(self.model.state_dict(),
                                                       self.config.model_save_name + ', epoch[{}]'.format(epoch),
                                                       'ACC', best_mcc)
                elif self.config.metric == 'MCC':
                    valid_mcc = valid_performance[4]
                    if valid_mcc > best_mcc:
                        best_mcc = valid_mcc
                        best_valid_performance = valid_performance
                    if self.config.save_best and valid_mcc > self.config.threshold:
                        self.IOManager.save_model_dict(self.model.state_dict(),
                                                       self.config.model_save_name + ', epoch[{}]'.format(epoch),
                                                       'MCC', best_mcc)
                else:
                    self.IOManager.log.Error('No Such Metric')
        return best_valid_performance

    def __SL_train(self, train_dataloader, test_dataloader):
        step = 0
        best_acc = 0
        best_mcc = 0
        best_performance = None

        for epoch in range(self.config.epoch):
            for batch in train_dataloader:
                data, label = batch
                if self.config.model == 'Transformer Encoder':
                    logits, _ = self.model(data)
                elif self.config.model == 'TextCNN':
                    logits = self.model(data)
                else:
                    logits = None
                    print('Mo Such Model')

                train_loss = self.__get_loss(logits, label)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                step += 1

                '''Periodic Train Log'''
                if step % self.config.interval_log == 0:
                    # pred = torch.max(logits, 1)[1]
                    # print('pred', pred)
                    corrects = (torch.max(logits, 1)[1] == label).sum()
                    the_batch_size = label.shape[0]
                    train_acc = 100.0 * corrects / the_batch_size
                    print('Epoch[{}] Batch[{}] - loss: {:.6f} | ACC: {:.4f}%({}/{})'.format(epoch, step, train_loss,
                                                                                            train_acc,
                                                                                            corrects,
                                                                                            the_batch_size))
                    self.visualizer.step_log_interval.append(step)
                    self.visualizer.train_metric_record.append(train_acc)
                    self.visualizer.train_loss_record.append(train_loss)

            '''Periodic Test'''
            if epoch % self.config.interval_test == 0:
                test_performance, avg_test_loss = self.__SL_test(test_dataloader)
                self.visualizer.step_test_interval.append(epoch)
                self.visualizer.test_metric_record.append(test_performance[0])
                self.visualizer.test_loss_record.append(avg_test_loss)
                self.test_performance.append(test_performance)

                log_text = '\n' + '=' * 20 + ' Test Performance. Epoch[{}] '.format(epoch) + '=' * 20 \
                           + '\n[ACC,\tSE,\t\tSP,\t\tAUC,\tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                    test_performance[0], test_performance[1], test_performance[2], test_performance[3],
                    test_performance[4]) \
                           + '\n' + '=' * 60
                self.IOManager.log.Info(log_text)

                # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
                if self.config.metric == 'ACC':
                    test_acc = test_performance[0]
                    if test_acc > best_acc:
                        best_acc = test_acc
                        best_performance = test_performance
                    if self.config.save_best and test_acc > self.config.threshold:
                        self.IOManager.save_model_dict(self.model.state_dict(),
                                                       self.config.model_save_name + ', epoch[{}]'.format(epoch),
                                                       'ACC', test_acc)
                elif self.config.metric == 'MCC':
                    test_mcc = test_performance[4]
                    if test_mcc > best_mcc:
                        best_mcc = test_mcc
                        best_performance = test_performance
                    if self.config.save_best and test_mcc > self.config.threshold:
                        self.IOManager.save_model_dict(self.model.state_dict(),
                                                       self.config.model_save_name + ', epoch[{}]'.format(epoch),
                                                       'MCC', test_mcc)
                else:
                    self.IOManager.log.Error('No Such Metric')

            if self.config.save_best and (epoch % 10 == 0):
                self.IOManager.save_model_dict(self.model.state_dict(), self.config.model_save_name,
                                               'Epoch', epoch)
        return best_performance

    def __SL_test(self, dataloader):
        corrects = 0
        test_batch_num = 0
        test_sample_num = 0
        avg_test_loss = 0
        pred_prob = []
        label_pred = []
        label_real = []

        with torch.no_grad():
            for batch in dataloader:
                data, label = batch
                if self.config.model == 'Transformer Encoder':
                    logits, _ = self.model(data)
                elif self.config.model == 'TextCNN':
                    logits = self.model(data)
                else:
                    logits = None
                    print('Mo Such Model')
                avg_test_loss += self.__get_loss(logits, label)

                pred_prob_all = F.softmax(logits, dim=1)  # 预测概率 [batch_size, class_num]
                pred_prob_positive = pred_prob_all[:, 1]  # 注意，极其容易出错
                pred_prob_sort = torch.max(pred_prob_all, 1)  # 每个样本中预测的最大的概率 [batch_size]
                pred_class = pred_prob_sort[1]  # 每个样本中预测的最大的概率所在的位置（类别） [batch_size]

                corrects += (pred_class == label).sum()
                test_sample_num += len(label)
                test_batch_num += 1
                pred_prob = pred_prob + pred_prob_positive.tolist()
                label_pred = label_pred + pred_class.tolist()
                label_real = label_real + label.tolist()

        avg_test_loss /= test_batch_num
        avg_acc = corrects.item() / test_sample_num
        print('Evaluation - loss: {:.6f}  ACC: {:.4f}%({}/{})'.format(avg_test_loss,
                                                                      avg_acc * 100,
                                                                      corrects,
                                                                      test_sample_num))
        self.avg_test_loss = avg_test_loss

        if self.config.num_class == 2:
            performance, ROC_data, PRC_data = self.__caculate_metric(pred_prob, label_pred, label_real)
        else:
            performance = [avg_acc, 0, 0, 0, 0]

        return performance, avg_test_loss

    def __meta_train(self, meta_train_tasks, meta_valid_tasks):
        best_meta_test_performance = None
        if self.config.model == 'ProtoNet':
            best_meta_test_performance = self.__train_ProtoNet(meta_train_tasks, meta_valid_tasks)
        elif self.config.model == 'MAML':
            best_meta_test_performance = self.__train_MAML(meta_train_tasks, meta_valid_tasks)
        else:
            self.IOManager.log.Error('No Such Meta Model')
        return best_meta_test_performance

    def __meta_test(self, meta_test_tasks, test_iteration, test_way, test_shot, test_query):
        test_ctr, avg_loss_sum, avg_loss_query_CE, avg_query_acc, avg_query_mi, avg_support_acc, avg_support_mi = 0, 0, 0, 0, 0, 0, 0

        # test_opt = None
        # if self.config.optimizer == 'AdamW':
        #     test_opt = torch.optim.AdamW(params=self.model.parameters(), lr=self.config.adapt_lr,
        #                                  weight_decay=self.config.reg)
        # elif self.config.optimizer == 'Adam':
        #     test_opt = torch.optim.Adam(params=self.model.parameters(), lr=self.config.adapt_lr,
        #                                 weight_decay=self.config.reg)
        # else:
        #     self.IOManager.log.Error('No Such Optimizer')

        for task_id in range(test_iteration):
            if not self.has_save_episodic_state_dict:
                self.episodic_state_dict = deepcopy(self.model.backbone.state_dict())
                self.has_save_episodic_state_dict = True

            self.model.backbone.load_state_dict(self.episodic_state_dict)

            test_task = self.dataManager.sample_task(meta_test_tasks)
            if test_task is None:
                meta_test_tasks = self.dataManager.reload_iterator()
                test_task = self.dataManager.sample_task(meta_test_tasks)

            if self.config.adapt_iteration != 0:
                for j in range(self.config.adapt_iteration):
                    if j % 5 == 0:
                        self.if_visual = True
                        # print('test_iteration[{}], adapt_iteration[{}]'.format(task_id, j))
                        self.model.config.title = '{} Adapt[{}]'.format(self.config.model_save_name, str(j))
                    else:
                        self.if_visual = False

                    loss_sum, loss_query_CE, support_mi, query_mi, \
                    support_acc, query_acc = self.model.fast_adapt(test_task,
                                                                   test_way,
                                                                   test_shot,
                                                                   test_query,
                                                                   self.config.if_transductive,
                                                                   False,
                                                                   self.if_visual)
                    self.optimizer.zero_grad()
                    loss_sum.backward()
                    self.optimizer.step()
                    # test_opt.zero_grad()
                    # loss_sum.backward()
                    # test_opt.step()
                    print(
                        '\ttask_id[{}] | j[{}] | loss_sum={:.3f} | loss_query_CE={:.3f} | query_acc={:.2f}% | query_mi={:.3f} | support_acc={:.2f}% | support_mi={:.3f}'.format(
                            task_id, j, loss_sum.item(), loss_query_CE.item(), query_acc.item() * 100, query_mi.item(),
                                                                               support_acc.item() * 100,
                            support_mi.item()))

            with torch.no_grad():
                loss_sum, loss_query_CE, support_mi, query_mi, support_acc, query_acc = self.model.fast_adapt(test_task,
                                                                                                              test_way,
                                                                                                              test_shot,
                                                                                                              test_query,
                                                                                                              self.config.if_transductive,
                                                                                                              False)
            print(
                'task_id[{}] | After Adapt | loss_sum={:.3f} | loss_query_CE={:.3f} | query_acc={:.2f}% | query_mi={:.3f} | support_acc={:.2f}% | support_mi={:.3f}'.format(
                    task_id, loss_sum.item(), loss_query_CE.item(), query_acc.item() * 100, query_mi.item(),
                                                                    support_acc.item() * 100, support_mi.item()))

            test_ctr += 1
            avg_loss_sum += loss_sum.item()
            avg_loss_query_CE += loss_query_CE.item()
            avg_query_acc += query_acc.item()
            avg_query_mi += query_mi.item()
            avg_support_acc += support_acc.item()
            avg_support_mi += support_mi.item()

            print(
                'Average | test_ctr[{}] | avg_loss_sum={:.3f} | avg_loss_query_CE={:.3f} | query_acc={:.2f}% | test_mi={:.3f} | support_acc={:.2f}% | support_mi={:.3f}'.format(
                    test_ctr, avg_loss_sum / test_ctr, avg_loss_query_CE / test_ctr,
                              100 * avg_query_acc / test_ctr,
                              avg_query_mi / test_ctr, 100 * avg_support_acc / test_ctr,
                              avg_support_mi / test_ctr))
            print('-' * 200)

        avg_performance = [avg_loss_sum / test_ctr, avg_loss_query_CE / test_ctr, avg_query_acc / test_ctr,
                           avg_query_mi / test_ctr, avg_support_acc / test_ctr, avg_support_mi / test_ctr]
        self.model.backbone.load_state_dict(self.episodic_state_dict)
        return avg_performance

    def __train_ProtoNet(self, meta_train_tasks, meta_valid_tasks):
        best_meta_test_performance = None
        best_acc = 0

        pbar = tqdm([i for i in range(self.config.max_epoch)])
        for epoch in pbar:
            self.model.train()
            if self.config.save_best and (epoch % 10 == 0):
                self.IOManager.save_model_dict(self.model.backbone.state_dict(), self.config.model_save_name,
                                               'Epoch', epoch)

            '''Train Episode'''
            train_ctr = 0
            train_loss = 0
            train_acc = 0
            train_mi = 0
            for task_id in range(1, self.config.meta_batch_size + 1):
                train_task = self.dataManager.sample_task(meta_train_tasks)

                for i in range(self.config.train_iteration):
                    if epoch % 5 == 0 and task_id == 10:
                        self.if_visual = True
                        self.model.config.title = '{} Epoch[{}]'.format(self.config.model_save_name, str(epoch))
                    else:
                        self.if_visual = False

                    loss_sum, loss_query_CE, support_mi, query_mi, support_acc, query_acc = self.model.fast_adapt(
                        train_task,
                        self.config.train_way,
                        self.config.train_shot,
                        self.config.train_query,
                        self.config.if_transductive,
                        True,
                        self.if_visual)
                    self.optimizer.zero_grad()
                    loss_sum.backward()
                    self.optimizer.step()

                train_ctr += 1
                train_loss += loss_sum.item()
                train_acc += query_acc.item()
                train_mi += query_mi.item()

            pbar.set_description('e[{}]'.format(epoch))
            pbar.set_postfix({'acc': '{0:.1f}%'.format(100 * train_acc / train_ctr),
                              'loss': '{0:.2f}'.format(train_loss / train_ctr),
                              'mi': '{0:.2f}'.format(train_mi / train_ctr)})
            print()
            self.visualizer.step_log_interval.append(epoch)
            self.visualizer.train_metric_record.append(train_acc / train_ctr)
            self.visualizer.train_loss_record.append(train_loss / train_ctr)

            # 保存临时模型参数
            self.episodic_state_dict = deepcopy(self.model.backbone.state_dict())
            self.has_save_episodic_state_dict = True
            '''Train Episode Over'''

            '''Validation Episode'''
            if epoch % self.config.valid_interval == 0 and epoch >= self.config.valid_start_epoch:
                self.model.eval()
                valid_performance = self.__meta_test(meta_valid_tasks, self.config.valid_iteration,
                                                     self.config.valid_way, self.config.valid_shot,
                                                     self.config.valid_query)

                self.visualizer.step_test_interval.append(epoch)
                self.visualizer.test_metric_record.append(valid_performance[2])
                # self.visualizer.test_loss_record.append(valid_performance[1])
                self.visualizer.test_loss_record.append(valid_performance[0])

                log_text = '\n' + '=' * 50 + ' Meta-Test Performance ' + '=' * 50 \
                           + '\navg_loss_sum={:.3f} | avg_loss_query_CE={:.3f} | query_acc={:.2f}% | test_mi={:.3f} | support_acc={:.2f}% | support_mi={:.3f}'.format(
                    valid_performance[0], valid_performance[1], 100 * valid_performance[2],
                    valid_performance[3], 100 * valid_performance[4], valid_performance[5]) \
                           + '\n' + '=' * 120
                self.IOManager.log.Info(log_text)

                self.meta_test_performance.append(valid_performance)
                valid_acc = valid_performance[2]
                if valid_acc >= best_acc:
                    best_acc = valid_acc
                    best_meta_test_performance = valid_performance

                if self.config.save_best and valid_acc > self.config.threshold:
                    self.IOManager.save_model_dict(self.model.backbone.state_dict(), self.config.model_save_name,
                                                   'ACC', valid_acc)

            if epoch % self.config.valid_draw == 0 and epoch >= self.config.valid_start_epoch and epoch != 0:
                self.visualizer.draw_train_test_curve()

        return best_meta_test_performance

    def __train_MAML(self, meta_train_tasks, meta_valid_tasks):
        pass

    def __meta_inference(self):
        test_ctr, avg_loss_sum, avg_loss_query_CE, avg_query_acc, avg_query_mi, avg_support_acc, avg_support_mi = 0, 0, 0, 0, 0, 0, 0
        avg_ACC, avg_SE, avg_SP, avg_AUC, avg_MCC = 0, 0, 0, 0, 0
        visual_prob = None

        for task_id in range(self.config.inference_iteration):
            if not self.has_save_episodic_state_dict:
                self.episodic_state_dict = deepcopy(self.model.backbone.state_dict())
                self.has_save_episodic_state_dict = True

            self.model.backbone.load_state_dict(self.episodic_state_dict)

            inference_task = self.dataManager.get_inference_task()

            best_mcc = 0
            best_performance = 0

            for j in range(self.config.adapt_iteration):
                loss_sum, loss_query_CE, support_mi, query_mi, \
                support_acc, query_acc, metric, query_probs = self.model.fast_adapt(inference_task,
                                                                                    self.config.inference_way,
                                                                                    self.config.inference_shot,
                                                                                    self.config.inference_query,
                                                                                    self.config.if_transductive,
                                                                                    False)
                self.optimizer.zero_grad()
                loss_sum.backward()
                self.optimizer.step()
                # metric: [ACC, Sensitivity, Specificity, AUC, MCC]

                if metric[4] > best_mcc:
                    best_mcc = metric[4]
                    best_performance = metric
                    visual_prob = query_probs

                print(
                    '\ttask_id[{}] | j[{}] | ACC: {:.4f} | SE: {:.4f} | SP: {:.4f} | AUC: {:.4f} | MCC: {:.4f}'.format(
                        task_id, j, metric[0], metric[1], metric[2], metric[3], metric[4]))

            with torch.no_grad():
                loss_sum, loss_query_CE, support_mi, query_mi, support_acc, query_acc, metric, query_probs = self.model.fast_adapt(
                    inference_task,
                    self.config.inference_way,
                    self.config.inference_shot,
                    self.config.inference_query,
                    self.config.if_transductive,
                    False)

            if metric[4] > best_mcc:
                best_mcc = metric[4]
                best_performance = metric
                visual_prob = query_probs

            print(
                'task_id[{}] | Best Performance | ACC: {:.4f} | SE: {:.4f} | SP: {:.4f} | AUC: {:.4f} | MCC: {:.4f}'.format(
                    task_id, best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                    best_performance[4]))

            test_ctr += 1

            # record the best evaluation
            avg_ACC += best_performance[0]
            avg_SE += best_performance[1]
            avg_SP += best_performance[2]
            avg_AUC += best_performance[3]
            avg_MCC += best_performance[4]

            print(
                'Average | test_ctr[{}] | ACC: {:.4f} | SE: {:.4f} | SP: {:.4f} | AUC: {:.4f} | MCC: {:.4f}'.format(
                    test_ctr, avg_ACC / test_ctr, avg_SE / test_ctr, avg_SP / test_ctr, avg_AUC / test_ctr,
                              avg_MCC / test_ctr))
            print('-' * 200)

        avg_performance = [avg_ACC / test_ctr, avg_SE / test_ctr, avg_SP / test_ctr, avg_AUC / test_ctr,
                           avg_MCC / test_ctr]
        self.model.backbone.load_state_dict(self.episodic_state_dict)

        log_text = '\n' + '=' * 50 + ' Final Meta-Test Performance ' + '=' * 50 \
                   + '\nACC: {:.4f} | SE: {:.4f} | SP: {:.4f} | AUC: {:.4f} | MCC: {:.4f}'.format(
            avg_performance[0], avg_performance[1], avg_performance[2],
            avg_performance[3], avg_performance[4]) \
                   + '\n' + '=' * 120
        self.IOManager.log.Info(log_text)

    def __few_shot_SL(self):
        best_performance = self.__SL_train(self.dataManager.train_dataloader, self.dataManager.test_dataloader)
        self.IOManager.log.Info('Best Performance: {}'.format(best_performance))
