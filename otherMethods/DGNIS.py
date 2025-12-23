import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
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

from Networks import Encoder_Causal, Classifier_CBGN, Encoder_Domain, DomainDiscriminator
from Networks import Decoder_CBGN
from calIndex import cal_index
from datasets.loadData import ReadMIMII, ReadMIMIIDG, MultiInfiniteDataLoader
import csv
from sklearn.model_selection import ParameterGrid
from torch.cuda.amp import GradScaler, autocast

import types


# ==============================================================================
# [新增] 封装类：将 Networks.py 中的组件适配为 DGNIS 需要的接口
# ==============================================================================

class FeatureGenerator_Inv(nn.Module):
    """
    不变量特征提取器
    使用 Encoder_Causal (InstanceNorm) 去除域信息
    """

    def __init__(self, configs):
        super().__init__()
        self.encoder = Encoder_Causal()

    def forward(self, x):
        # Encoder_Causal 返回 (f_map, f_vec)，我们只需要 f_vec (B, 4096)
        _, f_vec = self.encoder(x)
        return f_vec


class FeatureGenerator_Dom(nn.Module):
    """
    域特征提取器
    使用 Encoder_Domain (BatchNorm) 保留域信息
    """

    def __init__(self, configs):
        super().__init__()
        self.encoder = Encoder_Domain()

    def forward(self, x):
        # Encoder_Domain 返回 (f_map, f_vec)
        _, f_vec = self.encoder(x)
        return f_vec


class FaultClassifier_fan(nn.Module):
    """
    故障分类器封装
    """

    def __init__(self, configs):
        super().__init__()
        # input_dim=4096 对应 Encoder 输出的 256*4*4
        self.classifier = Classifier_CBGN(configs.num_classes, input_dim=4096)

    def forward(self, x):
        return self.classifier(x)


class DomainClassifier_fan(nn.Module):
    """
    域判别器封装
    """

    def __init__(self, configs):
        super().__init__()
        num_domains = len(configs.datasets_src)
        self.classifier = DomainDiscriminator(input_dim=4096, num_domains=num_domains)

    def forward(self, x):
        return self.classifier(x)


# ==============================================================================
# 工具函数
# ==============================================================================

def to_namespace(d: dict):
    if not isinstance(d, dict):
        return d
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = to_namespace(val)
        elif isinstance(val, (list, tuple)):
            d[key] = [to_namespace(x) if isinstance(x, dict) else x for x in val]
    return types.SimpleNamespace(**d)


with open(os.path.join(sys.path[0], 'config_files/DGNIS.yaml'), 'r', encoding='utf-8') as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = to_namespace(configs)
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            positive_mask = mask[i]
            negative_mask = (mask[i] == 0)
            if torch.sum(positive_mask) > 1 and torch.sum(negative_mask) > 0:
                dist_ap.append(dist[i][positive_mask].max().unsqueeze(0))
                dist_an.append(dist[i][negative_mask].min().unsqueeze(0))

        if not dist_ap or not dist_an:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


class CoralLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.num_domains = len(configs.datasets_src)

    def forward(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)
        return loss

    def cal_overall_coral_loss(self, features):
        loss = torch.tensor(0.0, device=self.device)
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                # 简单保护，防止切片越界
                start_i, end_i = i * self.batch_size, (i + 1) * self.batch_size
                start_j, end_j = j * self.batch_size, (j + 1) * self.batch_size
                if end_i <= features.size(0) and end_j <= features.size(0):
                    loss += self.forward(features[start_i:end_i], features[start_j:end_j])
        return loss


class DGNIS(nn.Module):
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
        self.margin = configs.margin

        self.use_domain_weight = configs.use_domain_weight
        self.domain_weight_scale = configs.domain_weight_scale
        self.num_domains = len(configs.datasets_src)

        self.fe_inv = FeatureGenerator_Inv(configs).to(self.device)  # 提取不变量
        self.fe_dom = FeatureGenerator_Dom(configs).to(self.device)  # 提取域特征
        self.dc = DomainClassifier_fan(configs).to(self.device)  # 域判别器

        # 创建多个故障分类器（每个域一个）
        self.fc1 = FaultClassifier_fan(configs).to(self.device)
        self.fc2 = FaultClassifier_fan(configs).to(self.device)
        self.fc3 = FaultClassifier_fan(configs).to(self.device)
        self.fc4 = FaultClassifier_fan(configs).to(self.device)
        self.fc5 = FaultClassifier_fan(configs).to(self.device)
        self.fc6 = FaultClassifier_fan(configs).to(self.device)
        self.fc7 = FaultClassifier_fan(configs).to(self.device)
        self.fc8 = FaultClassifier_fan(configs).to(self.device)

        self.fcs = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6, self.fc7, self.fc8]
        self.fcs = self.fcs[:self.num_domains]

        self.coral_loss = CoralLoss(configs)
        self.triplet_loss = TripletLoss(margin=self.margin)

        self.optimizer = torch.optim.Adam(params=list(self.parameters()), lr=self.lr)
        self.lbda_cr = configs.lbda_cr
        self.lbda_tp = configs.lbda_tp

        lr_list = [0.0001 / (1 + 10 * p / self.steps) ** 0.75 for p in range(self.steps + 1)]
        lambda_para = lambda step: lr_list[step] if step < len(lr_list) else lr_list[-1]
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_para)

        self.weight_step = None

    def update(self, minibatches):
        self.train()
        x = torch.cat([x for x, y in minibatches])
        labels = torch.cat([y for x, y in minibatches])
        x = x.to(self.device)
        labels = labels.to(self.device)

        fv_inv = self.fe_inv(x)

        loss_ce_total = 0
        loss_ce_list = []

        # 逐个域计算 CE Loss
        for i in range(self.num_domains):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size

            # 防止最后不足 batch 的情况导致索引越界
            if start_idx >= fv_inv.size(0):
                break
            real_end_idx = min(end_idx, fv_inv.size(0))

            fv_inv_i = fv_inv[start_idx:real_end_idx]
            labels_i = labels[start_idx:real_end_idx]

            if len(labels_i) == 0: continue

            logits_i = self.fcs[i](fv_inv_i)
            cross_entropy_i = F.cross_entropy(logits_i, labels_i)
            loss_ce_list.append(cross_entropy_i)
            loss_ce_total += cross_entropy_i

        # 域权重加权
        if self.use_domain_weight and len(loss_ce_list) > 0:
            if self.weight_step is None:
                self.weight_step = torch.ones(self.num_domains).to(self.device)
            else:
                ce_value_domains = torch.stack(loss_ce_list).to(self.device)
                weight_step = 1 + ce_value_domains / (loss_ce_total + 1e-8)
                self.weight_step = weight_step.to(self.device)

            loss_ce = 0
            for i in range(len(loss_ce_list)):
                loss_ce += self.weight_step[i] * loss_ce_list[i]
        else:
            loss_ce = loss_ce_total

        # CORAL Loss
        loss_cr = self.coral_loss.cal_overall_coral_loss(fv_inv)
        # Triplet Loss
        loss_tp = self.triplet_loss(fv_inv, labels)

        # 域分类 Loss
        domain_sizes = [x.shape[0] for x, _ in minibatches]
        dom_labels = torch.cat([
            torch.full((size,), i, dtype=torch.long)
            for i, size in enumerate(domain_sizes)
        ]).to(self.device)

        fv_dom = self.fe_dom(x)
        dom_logits = self.dc(fv_dom)
        loss_cd = F.cross_entropy(dom_logits, dom_labels)

        total_loss = loss_ce + self.lbda_cr * loss_cr + self.lbda_tp * loss_tp + loss_cd

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        loss = {}
        loss['ce'] = loss_ce.detach().cpu().data.numpy()
        loss['cr'] = loss_cr.detach().cpu().data.numpy()
        loss['tp'] = loss_tp.detach().cpu().data.numpy()
        loss['cd'] = loss_cd.detach().cpu().data.numpy()

        return loss

    def train_model(self, train_minibatches_iterator, test_loaders):
        self.to(self.device)
        loss_acc_result = {'loss_ce': [], 'loss_cr': [], 'loss_tp': [], 'loss_cd': [], 'acces': [], 'auc': []}

        print("train_model begin")
        for step in range(1, self.steps + 1):
            self.train()
            self.current_step = step
            try:
                minibatches_device = next(train_minibatches_iterator)
                losses = self.update(minibatches_device)

                loss_acc_result['loss_ce'].append(losses['ce'])
                loss_acc_result['loss_cr'].append(losses['cr'])
                loss_acc_result['loss_tp'].append(losses['tp'])
                loss_acc_result['loss_cd'].append(losses['cd'])

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
                    print("*" * 60)
            except StopIteration:
                print("Training iteration stopped early.")
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

            for _, (x, label_fault) in enumerate(the_loader):
                x = x.to(self.device)
                label_fault = label_fault.to(self.device)

                with torch.no_grad():
                    fv_inv = self.fe_inv(x)

                    logits_all = []
                    for fc in self.fcs:
                        logits_all.append(fc(fv_inv))

                    logits_all = torch.stack(logits_all, dim=1)

                    fv_dom = self.fe_dom(x)
                    domain_weights = self.dc(fv_dom)
                    domain_weights = F.softmax(domain_weights, dim=1).unsqueeze(2)

                    logits_weighted = logits_all * domain_weights
                    logits = logits_weighted.sum(dim=1)

                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                y_true.extend(label_fault.cpu().numpy())
                y_pred_labels.extend(preds.cpu().numpy())
                y_pred_probs.extend(probs.cpu().numpy())

            if len(y_true) > 0:
                y_true = np.array(y_true)
                y_pred_labels = np.array(y_pred_labels)
                y_pred_probs = np.array(y_pred_probs)
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

def main(seed, configs):
    """主函数（实验入口），支持两种数据集模式"""
    # =================================================================
    # ========== 1. Determine Mode and Define Domains =================
    # =================================================================

    # Check if we are in the new scenario mode based on the config value
    is_scenario_mode = str(configs.fan_section).startswith('s')

    datasets_src = []
    datasets_tgt = []

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


    # 初始化日志记录器
    currtime = str(time.time())[:10]  # 用时间戳创建唯一文件名

    model = DGNIS(configs)

    # 模型训练
    all_test_loaders = test_loaders_tgt + test_loaders_src
    valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

    # 执行训练
    loss_acc_result = model.train_model(
        train_minibatches_iterator,
        valid_test_loaders  # <-- 正确：传递清理后的列表
    )
    # 结果处理
    loss_acc_result = {
        'loss_ce': np.array(loss_acc_result['loss_ce']),
        'loss_cr': np.array(loss_acc_result['loss_cr']),
        'loss_tp': np.array(loss_acc_result['loss_tp']),
        'loss_cd': np.array(loss_acc_result['loss_cd']),
        'acces': np.array(loss_acc_result['acces']),
        'auc': np.array(loss_acc_result['auc']),  # ← 新增
    }

    print("===========================================================================================")
    print(f"best acc:{model.best_acc}, best auc:{model.best_auc}, best f1-score:{model.best_F1_score}")

    save_dir = os.path.join('result', 'DGNIS')  # 只定义目录路径
    filename = f'section0{configs.fan_section}_best_result.txt'  # 单独定义文件名
    file_path = os.path.join(save_dir, filename)  # 组合完整路径

    # 确保目录存在（仅创建目录）
    os.makedirs(save_dir, exist_ok=True)

    # 添加异常处理
    try:
        with open(file_path, 'a') as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            f.write(f"\nBest ACC: \n{model.best_acc:.4f}")
            f.write(f"\nBest AUC: \n{model.best_auc:.4f}")
            f.write(f"\nBest precision: \n{model.best_precision:.4f}\n")
            f.write(f"\nBest recall: \n{model.best_recall:.4f}\n")
            f.write(f"\nBest F1: \n{model.best_F1_score:.4f}\n")
    except PermissionError as pe:
        print(f"无法写入 {file_path}，错误详情：{str(pe)}")
        # 尝试备用路径（用户主目录）
        home_path = os.path.expanduser("~")
        backup_path = os.path.join(home_path, filename)
        print(f"尝试保存到备用路径：{backup_path}")
        with open(backup_path, 'a') as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            f.write(f"\nBest ACC: \n{model.best_acc:.4f}")
            f.write(f"\nBest AUC: \n{model.best_auc:.4f}")
            f.write(f"\nBest precision: \n{model.best_precision:.4f}\n")
            f.write(f"\nBest recall: \n{model.best_recall:.4f}\n")
            f.write(f"\nBest F1: \n{model.best_F1_score:.4f}\n")


if __name__ == '__main__':
    section_s = 's1','s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14'
    section = '00', '01', '02'
    sectionNumber = 3
    sectionsNumber = 14

    for i in range(sectionsNumber):
        run_times = 10
        configs.fan_section = section_s[i]
        for _ in range(run_times):
            print('----------------------------------------------------------', _)
            main( run_times, configs)
