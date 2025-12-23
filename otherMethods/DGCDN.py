import uuid
from fileinput import filename
import types
from sklearn.metrics import accuracy_score
import torch  # PyTorch深度学习框架核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from fontTools.merge.util import current_time
from fontTools.subset.svg import ranges
from numba.np.npdatetime import convert_datetime_for_arith
from sklearn.utils import compute_class_weight
from sympy import false
from sympy.physics.units import current
from torch.nn.functional import dropout
import torch.nn.functional as F  # 神经网络函数式接口
from torch.autograd import Function
# 导入科学计算和数据处理库
import numpy as np  # 数值计算库
import yaml  # YAML配置文件解析
import itertools  # 迭代工具
import copy  # 对象复制
import random  # 随机数生成
import os  # 操作系统接口
import time  # 时间相关功能
import sys  # 系统相关参数和函数
from builtins import object

import math  # 数学函数

# 导入自定义模块
from Networks import (  # CBGN网络组件
    Encoder_Domain, Encoder_Causal, Decoder_CBGN, Classifier_CBGN, DomainDiscriminator
)

from datasets.loadData import ReadMIMII, ReadMIMIIDG, MultiInfiniteDataLoader  # 风扇数据集加载器

from CompactBilinearPooling import CompactBilinearPooling

# 导入自定义工具类
from Logger import create_logger  # 日志创建器
from Report import GenReport  # 报告生成器
import csv
from sklearn.model_selection import ParameterGrid
from torch.cuda.amp import GradScaler, autocast
# from torch.amp import GradScaler, autocast

from calIndex import cal_index

import plot_tsne


def to_namespace(d: dict):
    """
    递归地将字典及其嵌套的字典转换为 types.SimpleNamespace。
    """
    if not isinstance(d, dict):
        return d

    # 遍历字典，递归地转换嵌套的字典或列表中的字典
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = to_namespace(val)
        elif isinstance(val, (list, tuple)):
            # 同样处理列表/元组中的字典
            d[key] = [to_namespace(x) if isinstance(x, dict) else x for x in val]

    return types.SimpleNamespace(**d)


# 加载配置文件
with open(os.path.join(sys.path[0], 'config_files/DGCDN.yaml'), 'r', encoding='utf-8') as f:
    '''从YAML文件加载配置参数'''
    configs_dict = yaml.load(f, Loader=yaml.FullLoader)
    configs = to_namespace(configs_dict)

    # 设置计算设备（GPU/CPU）
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'


class DGCDN(nn.Module):
    """条件域解耦生成网络（Conditional Domain Disentanglement Generative Network）"""

    def __init__(self, configs, seed, class_weights=None):
        super().__init__()
        self.current_step = None
        self.model_version = "v_CBGN_Arch"  # 标记版本
        self.configs = configs  # 配置参数
        self.device = torch.device(
            configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")
        self.dataset_type = configs.dataset_type
        self.seed = seed
        self.configs.seed = seed
        self.eps = configs.eps
        self.schedule = configs.schedule

        self.t_sne = True

        # 网络参数设置
        self.num_classes = configs.num_classes
        self.batch_size = configs.batch_size
        self.steps = configs.steps
        self.checkpoint_freq = configs.checkpoint_freq
        self.lr = configs.lr
        self.num_domains = len(configs.datasets_src)

        # Encoder_Causal 输出 (f_map, f_vec)，其中 f_vec 维度为 4096 (256*4*4)
        self.encoder_m = Encoder_Causal().to(self.device)
        self.encoder_h = Encoder_Causal().to(self.device)
        # Decoder_CBGN 输入维度为 512 (256*2)
        self.decoder = Decoder_CBGN().to(self.device)
        # Classifier_CBGN 需要指定输入维度 input_dim=4096
        self.classifer = Classifier_CBGN(self.num_classes, input_dim=4096).to(self.device)

        self.use_entropy_loss = configs.use_entropy_loss
        self.entropy_loss_weight = configs.entropy_loss_weight

        # 增加 Dropout 层
        self.dropout = nn.Dropout(p=configs.dropout)

        # 注意力机制超参数
        self.cbam_reduction = configs.cbam_reduction
        self.cbam_kernel_size = configs.cbam_kernel_size
        self.use_residual = configs.use_residual
        self.use_attention = True

        # CBAM 注意力：自动推断特征维度
        # Classifier_CBGN 的结构是 self.fc = nn.Sequential(...)
        # 我们查找其中的 Linear 层来获取输入维度
        channels = None
        for m in self.classifer.modules():
            if isinstance(m, nn.Linear):
                channels = m.in_features
                break
        if channels is None:
            channels = 4096
            # raise ValueError("无法从 Classifier 中推断特征维度")

        if self.use_attention:
            self.attention = CBAM1D(
                channels=channels,
                reduction=self.cbam_reduction,
                kernel_size=self.cbam_kernel_size,
                use_residual=self.use_residual
            ).to(self.device)  # 确保移动到设备
        else:
            self.attention = None

        self.focal_loss_gamma = configs.focal_loss_gamma
        # ===== 设置加权 FocalLoss =====
        if class_weights is not None:
            self.focal_loss = FocalLoss(gamma=self.focal_loss_gamma, weight=class_weights)
        else:
            self.focal_loss = FocalLoss(gamma=self.focal_loss_gamma)

        # 早停机制
        self.best_auc = -1
        self.best_acc = -1
        self.best_F1_score = -1
        self.best_recall = -1
        self.best_precision = -1
        self.early_stop_counter = 0
        self.early_stop = configs.early_stop
        self.best_model_path = None
        self.early_stopping_patience = configs.early_stopping_patience

        # 优化器设置
        self.optimizer = torch.optim.Adam(
            list(self.encoder_m.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder.parameters()) +
            list(self.classifer.parameters()),
            lr=self.lr,
            weight_decay=1e-4
        )

        # 仅在 CUDA 可用时启用 GradScaler
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()

        # 损失函数权重参数
        self.w_rc = configs.w_rc
        self.w_rr = configs.w_rr
        self.w_ca = configs.w_ca

        # 域权重相关参数
        self.weight_step = None
        self.use_domain_weight = configs.use_domain_weight

        # 学习率调度参数
        self.use_learning_rate_sheduler = configs.use_learning_rate_sheduler
        self.gamma = configs.gamma
        self.grad_clip = configs.grad_clip

    def forward_penul_fv(self, x):
        """获取倒数第二层健康特征向量（用于可视化）"""
        _, fh_vec = self.encoder_h(x)
        return fh_vec

    def forward_zd_fv(self, x):
        """获取机器域特征向量"""
        _, fm_vec = self.encoder_m(x)
        return fm_vec

    def adjust_learning_rate(self, step):
        lr = self.lr
        if not self.configs.cos:
            m = int(self.configs.schedule)
            if step % m == 0:
                lr *= self.gamma
        else:
            lr *= 0.5 * (1 + math.cos(math.pi * step / self.steps))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def cal_reconstruction_loss(self, x, x_rec):
        """计算信号重构损失（MSE）"""
        L = min(x.shape[2], x_rec.shape[2])
        x_cropped = x[:, :, :L]
        x_rec_cropped = x_rec[:, :, :L]
        return (x_rec_cropped - x_cropped).pow(2).mean()

    def cal_reduce_redundancy_loss(self, fm_vec, fh_vec):
        """计算特征冗余减少损失"""
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]

        # 特征归一化
        fm_vec = F.normalize(fm_vec, p=2, dim=1)
        fh_vec = F.normalize(fh_vec, p=2, dim=1)

        # 计算自相似矩阵
        sim_fm_vec = torch.matmul(fm_vec.T, fm_vec)
        sim_fh_vec = torch.matmul(fh_vec.T, fh_vec)

        E = torch.eye(D).to(self.device)
        denominator = torch.sum(1 - E) + float(self.eps)

        loss_fm = ((1 - E) * sim_fm_vec).pow(2).sum() / denominator
        loss_fh = ((1 - E) * sim_fh_vec).pow(2).sum() / denominator
        loss_fmh = torch.matmul(fh_vec.T, fm_vec).div(B).pow(2).mean()

        loss = loss_fm + loss_fh + loss_fmh
        return loss

    def cal_causal_aggregation_loss(self, fm_vec, fh_vec, labels, domain_labels):
        """改进后的因果聚合损失函数"""
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]

        fm_vec = F.normalize(fm_vec, p=2, dim=1, eps=float(self.eps))
        fh_vec = F.normalize(fh_vec, p=2, dim=1, eps=float(self.eps))

        labels = labels.contiguous().view(-1, 1)
        mask_fh = torch.eq(labels, labels.T).float().to(self.device)
        sim_fh = torch.mm(fh_vec, fh_vec.t()) / D

        pos_count = torch.sum(mask_fh) + float(self.eps)
        neg_count = torch.sum(1 - mask_fh) + float(self.eps)
        loss_fh = -(mask_fh * sim_fh).sum() / pos_count + ((1 - mask_fh) * sim_fh).sum() / neg_count

        domain_labels = domain_labels.contiguous().view(-1, 1)
        mask_fm = torch.eq(domain_labels, domain_labels.T).float().to(self.device)
        sim_fm = torch.mm(fm_vec, fm_vec.t()) / D

        pos_count_d = torch.sum(mask_fm) + float(self.eps)
        neg_count_d = torch.sum(1 - mask_fm) + float(self.eps)
        loss_fm = -(mask_fm * sim_fm).sum() / pos_count_d + ((1 - mask_fm) * sim_fm).sum() / neg_count_d

        total_loss = torch.clamp(loss_fh + loss_fm, -1e3, 1e3)
        return total_loss

    def update(self, minibatches):
        """改进的更新方法"""
        xs, ys, domain_labels = [], [], []

        for domain_idx, (x, y) in enumerate(minibatches):
            x = x.to(self.device)
            y = y.to(self.device)
            xs.append(x)
            ys.append(y)
            domain_labels.append(torch.full((x.size(0),), domain_idx, device=self.device))

        x = torch.cat(xs)
        y = torch.cat(ys)
        domain_labels = torch.cat(domain_labels)

        self.optimizer.zero_grad()

        with autocast(enabled=self.use_amp):
            output = self.forward(x, y, domain_labels)

            if self.use_entropy_loss:
                loss = self.w_rc * output['loss_rc'] + \
                       self.w_rr * output['loss_rr'] + \
                       self.w_ca * output['loss_ca'] + \
                       output['loss_cl'] + \
                       self.entropy_loss_weight * output['loss_entropy']
            else:
                loss = self.w_rc * output['loss_rc'] + \
                       self.w_rr * output['loss_rr'] + \
                       self.w_ca * output['loss_ca'] + \
                       output['loss_cl']

        if self.use_amp:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

        if self.use_entropy_loss:
            losses = {
                'rc': output['loss_rc'].detach().cpu().item(),
                'rr': output['loss_rr'].detach().cpu().item(),
                'ca': output['loss_ca'].detach().cpu().item(),
                'cl': output['loss_cl'].detach().cpu().item(),
                'loss_entropy': output['loss_entropy'].detach().cpu().item()
            }
        else:
            losses = {
                'rc': output['loss_rc'].detach().cpu().item(),
                'rr': output['loss_rr'].detach().cpu().item(),
                'ca': output['loss_ca'].detach().cpu().item(),
                'cl': output['loss_cl'].detach().cpu().item()
            }
        return losses

    def forward(self, x, labels, domain_labels=None):
        """前向传播过程"""
        output = {}
        B = x.shape[0]

        # 双编码器特征提取 (使用 CBGN Encoder_Causal)
        # Encoder_Causal 返回 (f_map, f_vec)
        fm_map, fm_vec = self.encoder_m(x)  # 机器特征
        fh_map, fh_vec = self.encoder_h(x)  # 健康特征

        # 特征融合与信号重构
        # CBGN Encoder 输出 map 通道为 256，拼接后为 512
        fmh_map = torch.cat([fm_map, fh_map], dim=1)
        x_rec = self.decoder(fmh_map)  # Decoder_CBGN 接受 512 通道

        # 在分类前对健康特征使用 Dropout
        fh_vec = self.dropout(fh_vec)

        # 如果启用了注意力机制
        if self.configs.use_attention and self.attention is not None:
            fh_vec = self.attention(fh_vec)

        # 健康状态分类
        logits = self.classifer(fh_vec)

        # 计算各项损失
        loss_rc = self.cal_reconstruction_loss(x, x_rec)
        loss_rr = self.cal_reduce_redundancy_loss(fm_vec, fh_vec)
        loss_ca = self.cal_causal_aggregation_loss(fm_vec, fh_vec, labels, domain_labels)

        loss_entropy = 0
        if self.use_entropy_loss:
            probs = torch.softmax(logits, dim=1)
            loss_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()

        # 动态域权重计算
        if self.use_domain_weight:
            ce_values = FocalLoss(self.focal_loss_gamma,
                                  weight=self.focal_loss.weight,
                                  reduction='none')(logits, labels)

            weight_list = []
            for d in range(self.num_domains):
                mask = (domain_labels == d)
                if mask.sum() > 0:
                    avg_loss = ce_values[mask].mean()
                else:
                    avg_loss = torch.tensor(0.0, device=self.device)
                weight_list.append(avg_loss)
            weight_step = torch.stack(weight_list)

            ce_value_sum = weight_step.mean() + float(self.eps)
            weight_step = 1 + (weight_step / ce_value_sum)
            self.weight_step = weight_step[domain_labels]
        else:
            self.weight_step = torch.ones(B, device=self.device)

        loss_cl = torch.mean(
            FocalLoss(self.focal_loss_gamma,
                      weight=self.focal_loss.weight,
                      reduction='none')(logits, labels)
            * self.weight_step
        )
        output.update({
            'loss_rc': loss_rc,
            'loss_rr': loss_rr,
            'loss_ca': loss_ca,
            'loss_cl': loss_cl,
            'fh_vec': fh_vec,
            'loss_entropy': loss_entropy
        })
        return output

    # 模型保存
    def save_checkpoint(self, current_time, step):
        if self.use_attention:
            save_dir = 'checkpoints\\section' + str(configs.fan_section)
        else:
            save_dir = 'checkpoints\\xiaorongshiyan\\without_attention\\' + 'section' + str(configs.fan_section)
        os.makedirs(save_dir, exist_ok=True)

        filename = f"section{configs.fan_section}_acc{self.best_acc:.4f}_auc{self.best_auc:.4f}_f1{self.best_F1_score:.4f}_{current_time}.pth"
        filename = os.path.join(save_dir, filename)

        checkpoint_data = {
            'encoder_m': self.encoder_m.state_dict(),
            'encoder_h': self.encoder_h.state_dict(),
            'decoder': self.decoder.state_dict(),
            'classifier': self.classifer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_F1': self.best_F1_score,
            'class_weights': self.focal_loss.weight,
            'seed': self.seed,
            'configs': self.configs
        }

        if self.configs.use_attention and self.attention is not None:
            checkpoint_data['attention'] = self.attention.state_dict()

        torch.save(checkpoint_data, filename)
        print(f"step:{step}, model is located at: {filename}")
        return filename


    def train_model(self, train_minibatches_iterator, test_loaders):
        """模型训练流程"""
        self.to(self.device)
        print("train_model begin")

        if self.use_entropy_loss:
            all_result = {'loss_rc': [], 'loss_rr': [], 'loss_ca': [], 'loss_cl': [], 'loss_entropy': [], 'acces': [],
                          'auc': [], 'f1-score': []}
        else:
            all_result = {'loss_rc': [], 'loss_rr': [], 'loss_ca': [], 'loss_cl': [], 'acces': [], 'auc': [],
                          'f1-score': []}
        current_time = time.strftime("%Y%m%d_%H%M%S")
        run_id = uuid.uuid4().hex[:8]

        for step in range(1, self.steps + 1):
            self.train()
            self.current_step = step

            try:
                minibatches_device = next(train_minibatches_iterator)
            except StopIteration:
                break

            losses = self.update(minibatches_device)
            all_result['loss_rc'].append(losses['rc'])
            all_result['loss_rr'].append(losses['rr'])
            all_result['loss_ca'].append(losses['ca'])
            all_result['loss_cl'].append(losses['cl'])

            if self.use_entropy_loss:
                all_result['loss_entropy'].append(losses['loss_entropy'])

            if self.use_learning_rate_sheduler:
                self.adjust_learning_rate(self.current_step)

            if step % self.checkpoint_freq == 0 or step == self.steps or step == 1:
                acc_results, auc_results, prec_results, recall_result, f1_results = self.test_model(test_loaders)
                all_result['acces'].append(acc_results)
                all_result.setdefault('auc', []).append(auc_results)

                print("*" * 60)
                print(f"  - step: {step}")

                avg_acc = np.mean(acc_results) if acc_results else -1.0
                avg_auc = np.mean(auc_results) if auc_results else -1.0
                avg_prec = np.mean(prec_results) if prec_results else -1.0
                avg_recall = np.mean(recall_result) if recall_result else -1.0
                avg_f1 = np.mean(f1_results) if f1_results else -1.0

                print(f"  - ACC: {avg_acc:.4f}")
                print(f"  - AUC: {avg_auc:.4f}")
                print(f"  - F1-Score: {avg_f1:.4f}")

                save_model_flag = False

                if avg_auc > self.best_auc:
                    self.best_auc = avg_auc
                    save_model_flag = True
                if avg_acc > self.best_acc:
                    self.best_acc = avg_acc
                    save_model_flag = True
                if avg_prec > self.best_precision:
                    self.best_precision = avg_prec
                    save_model_flag = True
                if avg_recall > self.best_recall:
                    self.best_recall = avg_recall
                    save_model_flag = True
                if avg_f1 > self.best_F1_score:
                    self.best_F1_score = avg_f1
                    save_model_flag = True

                if save_model_flag:
                    print("*" * 60)
                    print(f"New best model found based on AVERAGE metrics (Step: {self.current_step})")
                    print(f"Best AVG ACC: {self.best_acc:.4f}")
                    print(f"Best AVG AUC: {self.best_auc:.4f}")
                    print("*" * 60)
                    if self.best_model_path and os.path.exists(self.best_model_path):
                        try:
                            os.remove(self.best_model_path)
                        except:
                            pass
                    new_best_path = self.save_checkpoint(current_time, step)
                    self.best_model_path = new_best_path

                if self.early_stop:
                    if save_model_flag:
                        self.early_stop_counter = 0
                    else:
                        self.early_stop_counter += 1
                        if self.early_stop_counter >= self.early_stopping_patience:
                            print("Early stopping triggered!")
                            return all_result
        return all_result

    def test_model(self, loaders):
        self.eval()
        freeze_bn_stats(self)
        acc_results = []
        auc_results = []
        f1_results = []
        prec_results = []
        recall_result = []

        with torch.no_grad():
            for idx_loader, loader in enumerate(loaders):
                y_pred_lst = []
                y_prob_lst = []
                y_true_lst = []

                for x, label_fault in loader:
                    x = x.to(self.device)
                    label_fault = label_fault.to(self.device)

                    y_logits = self.classifer(self.encoder_h(x)[1])
                    y_probs = torch.softmax(y_logits, dim=1)
                    y_preds = torch.argmax(y_logits, dim=1)

                    y_prob_lst.append(y_probs.detach().cpu().numpy())
                    y_pred_lst.extend(y_preds.detach().cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                if not y_true_lst:
                    acc_results.append(0)
                    auc_results.append(0)
                    prec_results.append(0)
                    recall_result.append(0)
                    f1_results.append(0)
                    continue

                y_true = np.array(y_true_lst)
                y_pred = np.array(y_pred_lst)
                y_prob = np.vstack(y_prob_lst)

                acc_i, auc_i, prec_i, recall_i, f1_i = cal_index(y_true, y_pred, y_prob)

                acc_results.append(acc_i)
                auc_results.append(auc_i)
                prec_results.append(prec_i)
                recall_result.append(recall_i)
                f1_results.append(f1_i)

        self.train()

        t_sne = False
        if t_sne:
            pass

        return acc_results, auc_results, prec_results, recall_result, f1_results

    def predict(self, x):
        _, fh_vec = self.encoder_h(x)
        fh_vec = self.dropout(fh_vec)
        y_pred = self.classifer(fh_vec)
        return torch.max(y_pred, dim=1)[1]
def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
            m.eval()
class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention1D, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.BatchNorm1d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        B, C, L = x.size()
        avg_pool = torch.mean(x, dim=2)
        max_pool = torch.max(x, dim=2)[0]
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        att = torch.sigmoid(avg_out + max_out).view(B, C, 1)
        return x * att


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg_pool, max_pool], dim=1)
        att = self.sigmoid(self.conv(cat))
        return x * att


class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7, use_residual=False):
        super(CBAM1D, self).__init__()
        self.channel_att = ChannelAttention1D(channels, reduction)
        self.spatial_att = SpatialAttention1D(kernel_size)
        self.use_residual = use_residual

    def forward(self, x):
        # x: (B, C) or (B, C, L)
        is_vector = False
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (B, C, 1)
            is_vector = True

        out = self.channel_att(x)
        out = self.spatial_att(out)

        if self.use_residual:
            out = out + x

        if is_vector:
            out = out.squeeze(2)
        return out


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


def compute_class_weights_from_dataloader(loaders, num_classes):
    all_labels = []
    if not loaders:
        return torch.ones(num_classes, dtype=torch.float32)

    for loader in loaders:
        if loader is not None and len(loader.dataset) > 0:
            for _, labels in loader:
                all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        return torch.ones(num_classes, dtype=torch.float32)

    unique_labels = np.unique(all_labels)

    if len(unique_labels) < num_classes:
        weights = torch.ones(num_classes, dtype=torch.float32)
    else:
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),
            y=all_labels
        )
        weights = torch.tensor(weights, dtype=torch.float32)

    return weights


# ===========================================


def main(seed, configs):
    """主函数（实验入口），支持两种数据集模式"""

    # 检查当前是旧的section模式还是新的scenario模式
    is_scenario_mode = str(configs.fan_section).startswith('s')

    if is_scenario_mode:
        # =================================================================
        # ========== 1. 新的数据集加载逻辑 (Scenario-based) ==========
        # =================================================================
        scenario = configs.fan_section

        # 根据场景定义源域和目标域
        scenario_definitions = {
            # --- 原始场景 (3源, 1目标) ---
            's1': {'source': ['id_00', 'id_02', 'id_04'], 'target': ['id_06']},
            's2': {'source': ['id_00', 'id_02', 'id_06'], 'target': ['id_04']},
            's3': {'source': ['id_00', 'id_04', 'id_06'], 'target': ['id_02']},
            's4': {'source': ['id_02', 'id_04', 'id_06'], 'target': ['id_00']},
            # --- 新增场景 (2源, 2目标) ---
            's5': {'source': ['id_00', 'id_02'], 'target': ['id_04', 'id_06']},
            's6': {'source': ['id_00', 'id_04'], 'target': ['id_02', 'id_06']},
            's7': {'source': ['id_00', 'id_06'], 'target': ['id_02', 'id_04']},
            's8': {'source': ['id_02', 'id_04'], 'target': ['id_00', 'id_06']},
            's9': {'source': ['id_02', 'id_06'], 'target': ['id_00', 'id_04']},
            's10': {'source': ['id_04', 'id_06'], 'target': ['id_00', 'id_02']},
            # --- 新增场景 (1源, 3目标) ---
            's11': {'source': ['id_00'], 'target': ['id_02', 'id_04', 'id_06']},
            's12': {'source': ['id_02'], 'target': ['id_00', 'id_04', 'id_06']},
            's13': {'source': ['id_04'], 'target': ['id_00', 'id_02', 'id_06']},
            's14': {'source': ['id_06'], 'target': ['id_00', 'id_02', 'id_04']},
        }

        if scenario not in scenario_definitions:
            raise ValueError(f"未知的场景: {scenario}。请在 s1-s14 中选择。")

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

    # 更新配置对象（方便其他地方引用）
    configs.datasets_tgt = datasets_tgt
    configs.datasets_src = datasets_src

    # 创建训练和测试数据加载器
    train_test_loaders_src = [ds.load_dataloaders() for ds in datasets_object_src]
    train_loaders_src = [train for train, test in train_test_loaders_src if train is not None]
    test_loaders_src = [test for train, test in train_test_loaders_src if test is not None]

    # 自动计算类别权重
    class_weights = compute_class_weights_from_dataloader(train_loaders_src, configs.num_classes).to(configs.device)

    # 加载目标域数据加载器
    train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt if test is not None]

    # 创建跨域训练数据迭代器
    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)

    model = DGCDN(configs, seed, class_weights=class_weights)

    all_result = model.train_model(
        train_minibatches_iterator,
        test_loaders_tgt + test_loaders_src
    )

    all_result = {
        'loss_rc': np.array(all_result['loss_rc']),
        'loss_rr': np.array(all_result['loss_rr']),
        'loss_ca': np.array(all_result['loss_ca']),
        'loss_cl': np.array(all_result['loss_cl']),
        'acces': np.array(all_result['acces']),
        'auc': np.array(all_result['auc']),
    }

    if model.best_acc > 0:
        print(f"Run finished for seed {seed}. "
              f"Best ACC: {model.best_acc:.4f}, Best AUC: {model.best_auc:.4f}, "
              f"Best PRE: {model.best_precision:.4f}, Best REC: {model.best_recall:.4f}, "
              f"Best F1: {model.best_F1_score:.4f}")
        best_metrics = {
            'accuracy': model.best_acc,
            'auc': model.best_auc,
            'precision': model.best_precision,
            'recall': model.best_recall,
            'f1_score': model.best_F1_score
        }
        return best_metrics
    else:
        print(f"Run failed or did not yield a valid score for seed {seed}.")
        return {'accuracy': 0.0, 'auc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    param_grid = {
        'dropout': [0.3],
        'w_ca': [1],
        'w_rc': [1],
        'w_rr': [0.1],
    }

    scenarios_to_test = ['s1', '01', '02']
    N_SEEDS_PER_COMBO = 3
    output_csv_file = f'grid_search_full_metrics_{time.strftime("%Y%m%d%H%M%S")}.csv'

    param_keys = list(param_grid.keys())
    csv_headers = param_keys + ['scenario', 'seed', 'acc', 'auc', 'precision', 'recall',
                                'f1_score']

    grid = ParameterGrid(param_grid)
    print(f"Starting Grid Search. Total combinations: {len(grid)}")
    print(f"Results will be saved to: {output_csv_file}")

    with open(output_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    run_counter = 0
    for scenario in scenarios_to_test:
        print(f"\n{'=' * 30} TESTING SCENARIO: {scenario} {'=' * 30}")
        configs.fan_section = scenario

        for params in grid:
            for i in range(N_SEEDS_PER_COMBO):
                run_counter += 1
                seed = int(time.time()) + run_counter
                set_random_seed(seed)

                for key, value in params.items():
                    setattr(configs, key, value)

                print(f"\n--- [Run {run_counter}] Scenario: {scenario}, Seed: {seed}, Params: {params} ---")

                best_metrics_dict = main(seed, configs)

                acc_score = best_metrics_dict.get('accuracy', 0.0)
                auc_score = best_metrics_dict.get('auc', 0.0)
                precision_score = best_metrics_dict.get('precision', 0.0)
                recall_score = best_metrics_dict.get('recall', 0.0)
                f1_score = best_metrics_dict.get('f1_score', 0.0)

                result_row = list(params.values()) + [
                    scenario,
                    seed,
                    f"{acc_score:.4f}",
                    f"{auc_score:.4f}",
                    f"{precision_score:.4f}",
                    f"{recall_score:.4f}",
                    f"{f1_score:.4f}"
                ]

                with open(output_csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result_row)

                print(f"    Run finished. Result logged to {output_csv_file}")

    print(f"\n{'=' * 30} GRID SEARCH COMPLETE {'=' * 30}")
    print(f"All results have been saved to {output_csv_file}")