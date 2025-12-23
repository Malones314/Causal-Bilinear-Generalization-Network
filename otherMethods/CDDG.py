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
from Networks import Encoder_Causal, Classifier_CBGN, Encoder_Domain
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

class FocalLoss(nn.Module):
    def __init__(self, gamma, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        ce = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal_loss = (1 - pt) ** self.gamma * ce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
# 加载配置文件
with open(os.path.join(sys.path[0], 'config_files/CDDG.yaml'), 'r', encoding='utf-8') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = to_namespace(configs)

    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'


class CDDG(nn.Module):
    """条件域解耦生成网络"""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = torch.device(
            configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")
        self.dataset_type = configs.dataset_type

        self.num_classes = configs.num_classes
        self.batch_size = configs.batch_size
        self.steps = configs.steps
        self.checkpoint_freq = configs.checkpoint_freq
        self.lr = configs.lr
        self.num_domains = len(configs.datasets_src)

        # 根据数据集类型初始化网络组件
        self.encoder_m = Encoder_Domain().to(self.device)
        self.encoder_h = Encoder_Causal().to(self.device)
        self.decoder = Decoder_CBGN().to(self.device)
        self.classifer = Classifier_CBGN(self.num_classes, 4096).to(self.device)


        self.optimizer = torch.optim.Adam(
            list(self.encoder_m.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifer.parameters()),
            lr=self.lr
        )

        self.best_auc = -1
        self.best_acc = -1
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1

        self.w_rc = configs.w_rc
        self.w_rr = configs.w_rr
        self.w_ca = configs.w_ca

        self.weight_step = None
        self.use_domain_weight = configs.use_domain_weight

        self.use_learning_rate_sheduler = configs.use_learning_rate_sheduler
        self.gamma = configs.gamma
        self.grad_clip = configs.grad_clip

    def forward_penul_fv(self, x):
        _, fh_vec = self.encoder_h(x)
        fv = self.classifer.forward1(fh_vec)
        return fv

    def forward_zd_fv(self, x):
        _, fm_vec = self.encoder_m(x)
        return fm_vec

    def adjust_learning_rate(self, step):
        lr = self.lr
        if self.configs.cos:
            lr *= 0.5 * (1.0 + math.cos(math.pi * step / self.steps))
        else:
            for milestone in self.configs.schedule:
                lr *= self.gamma if step >= milestone else 1.0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def cal_reconstruction_loss(self, x, x_rec):
        L = min(x.shape[2], x_rec.shape[2])
        x_cropped = x[:, :, :L]
        x_rec_cropped = x_rec[:, :, :L]
        return (x_rec_cropped - x_cropped).pow(2).mean()

    def cal_reduce_redundancy_loss(self, fm_vec, fh_vec):
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]

        # 按样本维度归一化
        fm_vec = F.normalize(fm_vec, p=2, dim=1)
        fh_vec = F.normalize(fh_vec, p=2, dim=1)

        sim_fm_vec = torch.matmul(fm_vec.T, fm_vec)
        sim_fh_vec = torch.matmul(fh_vec.T, fh_vec)

        E = torch.eye(D).to(self.device)
        denominator = torch.sum(1 - E) + 1e-8

        loss_fm = ((1 - E) * sim_fm_vec).pow(2).sum() / denominator
        loss_fh = ((1 - E) * sim_fh_vec).pow(2).sum() / denominator
        loss_fmh = torch.matmul(fh_vec.T, fm_vec).div(B).pow(2).mean()

        loss = loss_fm + loss_fh + loss_fmh
        return loss

    def cal_causal_aggregation_loss(self, fm_vec, fh_vec, labels, domain_labels):
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]

        fm_vec = F.normalize(fm_vec, p=2, dim=1, eps=1e-8)
        fh_vec = F.normalize(fh_vec, p=2, dim=1, eps=1e-8)

        labels = labels.contiguous().view(-1, 1)
        mask_fh = torch.eq(labels, labels.T).float().to(self.device)
        sim_fh = torch.mm(fh_vec, fh_vec.t()) / D

        pos_count = torch.sum(mask_fh) + 1e-8
        neg_count = torch.sum(1 - mask_fh) + 1e-8
        loss_fh = -(mask_fh * sim_fh).sum() / pos_count + ((1 - mask_fh) * sim_fh).sum() / neg_count

        domain_labels = domain_labels.contiguous().view(-1, 1)
        mask_fm = torch.eq(domain_labels, domain_labels.T).float().to(self.device)
        sim_fm = torch.mm(fm_vec, fm_vec.t()) / D

        pos_count_d = torch.sum(mask_fm) + 1e-8
        neg_count_d = torch.sum(1 - mask_fm) + 1e-8
        loss_fm = -(mask_fm * sim_fm).sum() / pos_count_d + ((1 - mask_fm) * sim_fm).sum() / neg_count_d

        total_loss = torch.clamp(loss_fh + loss_fm, -1e3, 1e3)
        return total_loss

    def update(self, minibatches):
        xs, ys, domain_labels_list = [], [], []

        for domain_idx, (x, y) in enumerate(minibatches):
            x = x.to(self.device)
            y = y.to(self.device)
            xs.append(x)
            ys.append(y)
            domain_labels_list.append(torch.full((x.size(0),), domain_idx, dtype=torch.long, device=self.device))

        x = torch.cat(xs)
        y = torch.cat(ys)
        domain_labels = torch.cat(domain_labels_list)

        output = self.forward(x, y, domain_labels)

        logits = self.classifer(output['fh_vec'])

        # 使用Focal Loss
        loss_cl = FocalLoss(gamma=2.0)(logits, y)

        if self.use_domain_weight and self.weight_step is not None:
            loss_cl = (loss_cl * self.weight_step).mean()
        else:
            loss_cl = loss_cl.mean()

        loss = self.w_rc * output['loss_rc'] + \
               self.w_rr * output['loss_rr'] + \
               self.w_ca * output['loss_ca'] + \
               loss_cl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        losses = {
            'rc': output['loss_rc'].detach().cpu().item(),
            'rr': output['loss_rr'].detach().cpu().item(),
            'ca': output['loss_ca'].detach().cpu().item(),
            'cl': loss_cl.detach().cpu().item()
        }
        return losses

    def forward(self, x, labels, domain_labels=None):
        output = {}
        B = x.shape[0]

        if domain_labels is None:
            # 默认假设
            domain_labels = torch.zeros(B, dtype=torch.long).to(self.device)

        fm_map, fm_vec = self.encoder_m(x)
        fh_map, fh_vec = self.encoder_h(x)

        fmh_map = torch.cat([fm_map, fh_map], dim=1)
        x_rec = self.decoder(fmh_map)

        logits = self.classifer(fh_vec)

        loss_rc = self.cal_reconstruction_loss(x, x_rec)
        loss_rr = self.cal_reduce_redundancy_loss(fm_vec, fh_vec)
        loss_ca = self.cal_causal_aggregation_loss(fm_vec, fh_vec, labels, domain_labels)

        if self.use_domain_weight:
            # 计算CrossEntropy用于权重调整
            ce_values = F.cross_entropy(logits, labels, reduction='none')
            weight_list = []
            # domain_labels 是 0, 1, ..., num_domains-1
            for d in range(self.num_domains):
                mask = (domain_labels == d)
                if mask.sum() > 0:
                    avg_loss = ce_values[mask].mean()
                else:
                    avg_loss = torch.tensor(0.0, device=self.device)
                weight_list.append(avg_loss)

            weight_step_domains = torch.stack(weight_list)
            ce_value_sum = weight_step_domains.sum() + 1e-8
            weight_factors = 1 + weight_step_domains / ce_value_sum

            # 映射回每个样本
            self.weight_step = weight_factors[domain_labels]
        else:
            self.weight_step = torch.ones(B).to(self.device)

        output.update({
            'loss_rc': loss_rc,
            'loss_rr': loss_rr,
            'loss_ca': loss_ca,
            'fh_vec': fh_vec
        })
        return output

    def train_model(self, train_minibatches_iterator, test_loaders):
        self.to(self.device)
        print("train_model begin")
        loss_acc_result = {'loss_rc': [], 'loss_rr': [], 'loss_ca': [], 'loss_cl': [], 'acces': [], 'auc': []}

        for step in range(1, self.steps + 1):
            self.train()
            self.current_step = step

            try:
                minibatches_device = next(train_minibatches_iterator)
                losses = self.update(minibatches_device)

                if self.use_learning_rate_sheduler:
                    self.adjust_learning_rate(self.current_step)

                loss_acc_result['loss_rc'].append(losses['rc'])
                loss_acc_result['loss_rr'].append(losses['rr'])
                loss_acc_result['loss_ca'].append(losses['ca'])
                loss_acc_result['loss_cl'].append(losses['cl'])

                if step % self.checkpoint_freq == 0 or step == self.steps:
                    print(f"Step: {step}")
                    acc_results, auc_results, prec_results, recall_result, f1_results = self.test_model(test_loaders)
                    loss_acc_result['acces'].append(acc_results)
                    loss_acc_result['auc'].append(auc_results)

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
                print("Training iteration stopped early.")
                break

        return loss_acc_result

    def test_model(self, loaders):
        self.eval()
        acc_results = []
        auc_results = []
        f1_results = []
        prec_results = []
        recall_result = []
        with torch.no_grad():
            for loader in loaders:
                y_pred_lst = []
                y_prob_lst = []
                y_true_lst = []
                for x, label_fault in loader:
                    x = x.to(self.device)
                    label_fault = label_fault.to(self.device)
                    _, fh_vec = self.encoder_h(x)
                    y_logits = self.classifer(fh_vec)
                    y_probs = torch.softmax(y_logits, dim=1)
                    y_preds = torch.argmax(y_logits, dim=1)

                    y_prob_lst.append(y_probs.detach().cpu().numpy())
                    y_pred_lst.extend(y_preds.detach().cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                if len(y_true_lst) > 0:
                    y_true = np.array(y_true_lst)
                    y_pred = np.array(y_pred_lst)
                    y_prob = np.vstack(y_prob_lst)
                    acc_i, auc_i, prec_i, recall_i, f1_i = cal_index(y_true, y_pred, y_prob)
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
        _, fh_vec = self.encoder_h(x)
        y_pred = self.classifer(fh_vec)
        return torch.max(y_pred, dim=1)[1]


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

    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)

    # Run once per call (or loop here if preferred)
    model = CDDG(configs)

    all_test_loaders = test_loaders_tgt + test_loaders_src
    valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

    loss_acc_result = model.train_model(
        train_minibatches_iterator,
        valid_test_loaders
    )

    print("===========================================================================================")
    print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

    save_dir = os.path.join('result', 'CDDG')
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
        print(f"Save failed: {e}")


if __name__ == '__main__':
    # 示例运行逻辑
    section_s = '00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'
    section = '00', '01', '02'
    sectionNumber = 3
    sectionsNumber = 17

    # 调试运行：只运行一次
    for i in range(sectionsNumber):  # range(sectionsNumber)
        run_times = 1
        configs.fan_section = section_s[i]
        for _ in range(run_times):
            print('----------------------------------------------------------', _)
            main(run_times, configs)