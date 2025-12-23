import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
# from torch.autograd import Function
# from torchvision import models
from torchsummary import summary

import numpy as np
import yaml
import itertools
import copy
import random
import os
import time
import sys
from builtins import object
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, freqz, hilbert
import pickle
import matplotlib.pyplot as plt
import math

from Conv2dBlock import Conv2dBlock
from Networks import Encoder_Causal, Classifier_CBGN
from Networks import Decoder_CBGN
from calIndex import cal_index
from datasets.loadData import ReadMIMII, ReadMIMIIDG, MultiInfiniteDataLoader
# from Report import GenReport  # 报告生成器 (如果不需要可注释)
import csv
from sklearn.model_selection import ParameterGrid
from torch.cuda.amp import GradScaler, autocast

import types


def to_namespace(d: dict):
    """
    递归地将字典及其嵌套的字典转换为 types.SimpleNamespace。
    """
    if not isinstance(d, dict):
        return d
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = to_namespace(val)
        elif isinstance(val, (list, tuple)):
            d[key] = [to_namespace(x) if isinstance(x, dict) else x for x in val]
    return types.SimpleNamespace(**d)


with open(os.path.join(sys.path[0], 'config_files/CCN.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = to_namespace(configs)

    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'


class CausalConsistencyLoss(nn.Module):
    '''
    Causal Consistency Loss
    '''

    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes
        self.device = configs.device
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6).to(self.device)

    def cal_cos_dis(self, x, y):
        cos_xy = self.cos_sim(x, y)
        loss = 1 - abs(cos_xy)
        return loss

    def forward(self, flatten_features, data_label, weight):
        # flatten_features 应该是 (Batch, Feature_Dim)，例如 (Batch, 4096)
        list_category = [[] for i in range(self.num_classes)]
        list_category_weight = [[] for i in range(self.num_classes)]

        for i, fv, w in zip(data_label, flatten_features, weight):
            # fv 是 (4096,)，reshape 为 (1, 4096)
            fv = torch.reshape(fv, (1, fv.size(0)))
            list_category[i].append(fv)
            list_category_weight[i].append(w)

        total_cc_loss = 0
        for i in range(self.num_classes):
            if len(list_category[i]) > 0:
                fm_i = torch.cat(tuple(list_category[i]), dim=0).to(self.device)
                w_i = torch.tensor(list_category_weight[i]).to(self.device)
                fm_i_mean = torch.mean(fm_i, dim=0, keepdim=True).to(self.device)

                cc_loss_i = torch.sum(self.cal_cos_dis(fm_i_mean, fm_i) * w_i).to(self.device)
                total_cc_loss = total_cc_loss + cc_loss_i

        total_cc_loss = total_cc_loss / flatten_features.size(0)

        return total_cc_loss


class CollaborativeTrainingLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes
        self.device = configs.device
        self.ce_loss = nn.CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, logits, data_label, weight):
        list_category = [[] for i in range(self.num_classes)]
        list_category_weight = [[] for i in range(self.num_classes)]
        for i, fv, w in zip(data_label, logits, weight):
            fv = torch.reshape(fv, (1, fv.size(0)))
            list_category[i].append(fv)
            list_category_weight[i].append(w)

        total_ct_loss = 0
        for i in range(self.num_classes):
            if len(list_category[i]) > 0:
                logits_i = torch.cat(tuple(list_category[i]), dim=0).to(self.device)
                w_i = torch.tensor(list_category_weight[i]).to(self.device)
                label_i = torch.tensor([i] * logits_i.size(0)).to(self.device)
                ce_i = self.ce_loss(logits_i, label_i)
                ce_i_mean = torch.mean(ce_i).to(self.device)

                ct_loss_i = torch.sum((ce_i - ce_i_mean).pow(2) * w_i).to(self.device)
                total_ct_loss = total_ct_loss + ct_loss_i

        total_ct_loss = torch.sqrt(total_ct_loss / logits.size(0)).to(self.device)

        return total_ct_loss


class CCN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.best_auc = -1
        self.best_acc = -1
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1
        self.configs = configs
        self.device = configs.device
        self.dataset_type = configs.dataset_type
        self.checkpoint_freq = configs.checkpoint_freq
        self.steps = configs.steps
        self.lr = configs.lr
        self.batch_size = configs.batch_size
        self.use_domain_weight = configs.use_domain_weight

        # 拆分为 encoder 和 classifier
        # 注意：Encoder_Causal 定义中没有参数，所以去掉 configs
        self.encoder = Encoder_Causal().to(self.device)
        # 4096 是 Encoder_Causal 输出的特征维度 (256*4*4)
        self.classifier = Classifier_CBGN(configs.num_classes, 4096).to(self.device)

        # 优化器包含两部分的参数
        self.optimizer = torch.optim.Adagrad(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.lr
        )

        zz = [1] * 25 + [0.5] * 150 + [0.1] * 400
        self.lambda_func = lambda step: zz[step] if step < len(zz) else 0.1  # 防止索引越界
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self.lambda_func)

        self.cc_loss = CausalConsistencyLoss(configs)
        self.ct_loss = CollaborativeTrainingLoss(configs)
        self.cl_loss = nn.CrossEntropyLoss(reduction='none')

        self.lbda_cc = configs.lbda_cc
        self.lbda_ct = configs.lbda_ct
        self.num_domains = len(configs.datasets_src)
        self.weight_step = None

    def update(self, minibatches):
        x = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])
        x = x.to(self.device)
        labels = labels.to(self.device)

        # 分步前向传播
        f_map, f_vec = self.encoder(x)
        logits = self.classifier(f_vec)  # 使用 classifier 得到 logits

        if self.weight_step is None:
            self.weight_step = torch.ones(x.shape[0]).to(self.device)
        else:
            ce_values = self.cl_loss(logits, labels)
            batch_size_total = ce_values.shape[0]
            batch_sizes = [x.shape[0] for x, _ in minibatches]

            ce_values_2d = torch.split(ce_values, batch_sizes)
            ce_value_domain = torch.tensor([v.mean().item() for v in ce_values_2d]).to(self.device)
            ce_value_sum = torch.sum(ce_value_domain)
            weight_step = 1 + ce_value_domain / (ce_value_sum + 1e-8)  # 加 epsilon 防止除零

            self.weight_step = torch.cat([
                weight_step[i].repeat(batch_sizes[i]) for i in range(self.num_domains)
            ]).to(self.device)

        # 传入 f_vec (展平特征) 而不是 f_map
        cc_loss = self.cc_loss(f_vec, labels, self.weight_step)
        ct_loss = self.ct_loss(logits, labels, self.weight_step)

        cl_loss = torch.mean(self.cl_loss(logits, labels) * self.weight_step)

        total_loss = cl_loss + self.lbda_cc * cc_loss + self.lbda_ct * ct_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss_cc = cc_loss.detach().cpu().data.numpy()
        loss_ct = ct_loss.detach().cpu().data.numpy()
        loss_cl = cl_loss.detach().cpu().data.numpy()

        losses = {}
        losses['cc'] = loss_cc
        losses['ct'] = loss_ct
        losses['cl'] = loss_cl

        return losses

    def train_model(self, train_minibatches_iterator, test_loaders):
        self.to(self.device)

        loss_acc_result = {'loss_cc': [], 'loss_ct': [], 'loss_cl': [], 'acces': []}

        for step in range(1, self.steps + 1):
            self.train()
            # self.current_step = step # 似乎未使用
            try:
                minibatches_device = next(train_minibatches_iterator)
                losses = self.update(minibatches_device)

                loss_acc_result['loss_cc'].append(losses['cc'])
                loss_acc_result['loss_ct'].append(losses['ct'])
                loss_acc_result['loss_cl'].append(losses['cl'])

                if step % self.checkpoint_freq == 0 or step == self.steps:
                    acc_results, auc_results, prec_results, recall_result, f1_results = self.test_model(test_loaders)
                    loss_acc_result['acces'].append(acc_results)

                    print(f"Step: {step}")
                    if auc_results[0] > self.best_auc:
                        print("auc_result[0]:", auc_results[0])
                        self.best_auc = auc_results[0]
                    if acc_results[0] > self.best_acc:
                        print("acc_result[0]:", acc_results[0])
                        self.best_acc = acc_results[0]
                    if f1_results[0] > self.best_F1_score:
                        print("f1_results[0]:", f1_results[0])
                        self.best_F1_score = f1_results[0]
                    if prec_results[0] > self.best_precision:
                        print("prec_results[0]:", prec_results[0])
                        self.best_precision = prec_results[0]
                    if (recall_result[0] > self.best_recall) and (recall_result[0] < 1):
                        self.best_recall = recall_result[0]
                        print("best recall:", self.best_recall)
                    print("*" * 60)
            except StopIteration:
                print("Training data exhausted.")
                break

        return loss_acc_result

    def test_model(self, loaders):
        self.eval()
        num_loaders = len(loaders)
        acc_results = []
        auc_results = []
        f1_results = []
        prec_results = []
        recall_result = []
        for i in range(num_loaders):
            the_loader = loaders[i]
            y_pred_labels = []
            y_true = []
            y_pred_probs = []
            for j, batched_data in enumerate(the_loader):
                x, label_fault = batched_data
                x = x.to(self.device)
                label_fault = label_fault.to(self.device)

                with torch.no_grad():
                    # 使用 encoder + classifier
                    _, f_vec = self.encoder(x)
                    logits = self.classifier(f_vec)
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                y_pred_labels.extend(preds.cpu().numpy())
                y_true.extend(label_fault.cpu().numpy())
                y_pred_probs.extend(probs.cpu().numpy())

            y_true = np.array(y_true)
            y_pred_labels = np.array(y_pred_labels)
            y_pred_probs = np.array(y_pred_probs)

            # 增加检查防止空数据报错
            if len(y_true) > 0:
                acc_i, auc_i, prec_i, recall_i, f1_i = cal_index(y_true, y_pred_labels, y_pred_probs)
            else:
                acc_i, auc_i, prec_i, recall_i, f1_i = 0, 0, 0, 0, 0

            acc_results.append(acc_i)
            auc_results.append(auc_i)
            prec_results.append(prec_i)
            recall_result.append(recall_i)
            f1_results.append(f1_i)

        self.train()
        return acc_results, auc_results, prec_results, recall_result, f1_results

    def predict(self, x):
        _, f_vec = self.encoder(x)
        logits = self.classifier(f_vec)
        return torch.max(logits, dim=1)[1]


def main(seed, configs):
    # =================================================================
    # ========== 1. Determine Mode and Define Domains =================
    # =================================================================

    is_scenario_mode = str(configs.fan_section).startswith('s')


    if is_scenario_mode:
        scenario = configs.fan_section
        scenario_definitions = {
            's1': {'source': ['id_00', 'id_02', 'id_04'], 'target': ['id_06']},
            's2': {'source': ['id_00', 'id_02', 'id_06'], 'target': ['id_04']},
            's3': {'source': ['id_00', 'id_04', 'id_06'], 'target': ['id_02']},
            's4': {'source': ['id_02', 'id_04', 'id_06'], 'target': ['id_00']},
            's5': {'source': ['id_00', 'id_02'], 'target': ['id_04', 'id_06']},
            's6': {'source': ['id_00', 'id_04'], 'target': ['id_02', 'id_06']},
            's7': {'source': ['id_00', 'id_06'], 'target': ['id_02', 'id_04']},
            's8': {'source': ['id_02', 'id_04'], 'target': ['id_00', 'id_06']},
            's9': {'source': ['id_02', 'id_06'], 'target': ['id_00', 'id_04']},
            's10': {'source': ['id_04', 'id_06'], 'target': ['id_00', 'id_02']},
            's11': {'source': ['id_00'], 'target': ['id_02', 'id_04', 'id_06']},
            's12': {'source': ['id_02'], 'target': ['id_00', 'id_04', 'id_06']},
            's13': {'source': ['id_04'], 'target': ['id_00', 'id_02', 'id_06']},
            's14': {'source': ['id_06'], 'target': ['id_00', 'id_02', 'id_04']},
        }
        if scenario not in scenario_definitions:
            raise ValueError(f"未知的场景: {scenario}")

        datasets_src = scenario_definitions[scenario]['source']
        datasets_tgt = scenario_definitions[scenario]['target']

        datasets_object_src = [ReadMIMII(scenario, domain_id, seed, configs) for domain_id in datasets_src]
        datasets_object_tgt = [ReadMIMII(scenario, domain_id, seed, configs) for domain_id in datasets_tgt]

    else:
        section = str(configs.fan_section).zfill(2)
        if section == '00':
            datasets_src = ['W', 'X']
            datasets_tgt = ['Y', 'Z']
        elif section == '01':
            datasets_src = ['A', 'B']
            datasets_tgt = ['C']
        elif section == '02':
            datasets_src = ['L1', 'L2']
            datasets_tgt = ['L3', 'L4']
        else:
            raise ValueError(f"未知的 Section: {section}")

        datasets_object_src = [ReadMIMIIDG(domain, seed, section, configs) for domain in datasets_src]
        datasets_object_tgt = [ReadMIMIIDG(domain, seed, section, configs) for domain in datasets_tgt]

    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src

    train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
    train_loaders_src = [train for train, test in train_test_loaders_src if train is not None]
    test_loaders_src = [test for train, test in train_test_loaders_src if test is not None]

    train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt if test is not None]

    if not train_loaders_src:
        print("错误：没有可用的训练数据加载器！请检查数据文件路径和内容。")
        return

    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)

    model = CCN(configs)  # 不在这里循环，Model重建放在循环外或内均可，但这里只运行一次训练

    all_test_loaders = test_loaders_tgt + test_loaders_src
    valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

    loss_acc_result = model.train_model(
        train_minibatches_iterator,
        valid_test_loaders,
    )

    # 结果保存逻辑保持不变
    print("===========================================================================================")
    print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

    save_dir = os.path.join('result', 'CCN')
    filename = f'section0{configs.fan_section}_best_result.txt'
    file_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    try:
        with open(file_path, 'a') as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            f.write(f"\nBest ACC: \n{model.best_acc:.4f}")
            f.write(f"\nBest AUC: \n{model.best_auc:.4f}")
            f.write(f"\nBest precision: \n{model.best_precision:.4f}\n")
            f.write(f"\nBest recall: \n{model.best_recall:.4f}\n")
            f.write(f"\nBest F1: \n{model.best_F1_score:.4f}\n")
    except Exception as e:
        print(f"保存结果失败: {e}")


if __name__ == '__main__':
    # 保持原有的执行逻辑
    section_s = '00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'
    section = '00', '01', '02'
    sectionNumber = 3
    sectionsNumber = 17  # 你可以根据需要调整这个数字，比如只跑前3个

    # 为了调试方便，这里只演示跑 section 00 (即 i=0)
    # 实际运行时请改回 range(sectionsNumber)
    for i in range(sectionsNumber):  # range(sectionsNumber)
        run_times = 10
        configs.fan_section = section_s[i]
        for _ in range(run_times):
            print('----------------------------------------------------------', _)
            main(run_times, configs)