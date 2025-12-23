import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import yaml
import random
from math import log2
import types

from calIndex import cal_index
from datasets.loadData import ReadMIMII, ReadMIMIIDG, MultiInfiniteDataLoader


# ==============================================================================
# 模型组件
# ==============================================================================

class ECA(nn.Module):
    """ 高效通道注意力 (ECA) 模块 """

    def __init__(self, channels, b=1, y=2):
        super(ECA, self).__init__()
        t = int(abs((log2(channels) + b) / y))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y).squeeze(1).unsqueeze(-1)
        return x * y.expand_as(x)


class MaximizedAggregationRouting(nn.Module):
    """ 最大化聚合路由算法 """

    def __init__(self, in_caps, out_caps, in_dim, out_dim, num_iterations=3):
        super(MaximizedAggregationRouting, self).__init__()
        self.num_iterations = num_iterations
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W_A = nn.Parameter(torch.randn(in_caps, in_dim))
        self.B_A = nn.Parameter(torch.randn(in_caps))
        self.W_F1 = nn.Parameter(torch.randn(in_caps, in_dim, in_dim))
        self.W_F2 = nn.Parameter(torch.randn(out_caps, in_dim, out_dim))
        self.B_F2 = nn.Parameter(torch.randn(out_caps, out_dim))
        self.W_G1 = nn.Parameter(torch.randn(out_dim, out_dim))
        self.W_G2 = nn.Parameter(torch.randn(out_caps, out_dim, in_dim))
        self.B_G2 = nn.Parameter(torch.randn(out_caps, in_dim))
        self.layer_norm = nn.LayerNorm(out_dim)
        self.W_S = nn.Parameter(torch.randn(in_caps, out_caps, in_dim))
        self.B_S = nn.Parameter(torch.randn(in_caps, out_caps))
        self.beta_use = nn.Parameter(torch.randn(in_caps, out_caps))
        self.beta_ign = nn.Parameter(torch.randn(in_caps, out_caps))

    def forward(self, x_inp):
        B = x_inp.shape[0]
        a_e = torch.einsum('bif,if->bi', x_inp, self.W_A) / np.sqrt(self.in_dim) + self.B_A
        R_em = torch.ones(B, self.in_caps, self.out_caps, device=x_inp.device) / self.out_caps
        for r in range(self.num_iterations):
            f_a_e = torch.sigmoid(a_e)
            D_use = f_a_e.unsqueeze(-1) * R_em
            D_ign = f_a_e.unsqueeze(-1) - D_use
            phi_em = self.beta_use * D_use - self.beta_ign * D_ign
            temp_F = torch.einsum('bif,ifh->bih', x_inp, self.W_F1) / np.sqrt(self.in_dim)
            temp_F_routed = torch.einsum('bi,bih->bih', phi_em.sum(dim=-1), temp_F)
            x_out = torch.einsum('bih,mhd->bmd', temp_F_routed, self.W_F2) + self.B_F2
            x_out_norm = self.layer_norm(x_out)
            temp_G = torch.einsum('bmd,dh->bmh', x_out_norm, self.W_G1)
            x_inp_hat = torch.einsum('bmh,mhi->bmi', temp_G, self.W_G2) + self.B_G2
            consistency = torch.einsum('bif,bmf,imf->bim', x_inp, x_inp_hat, self.W_S) + self.B_S
            S_em = F.log_softmax(consistency, dim=-1)
            R_em = torch.exp(S_em)
        return x_out


class MAACCN(nn.Module):
    """ 论文核心模型 MAACCN """

    def __init__(self, configs):
        super(MAACCN, self).__init__()
        self.configs = configs
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")

        # --- 1. One-Dimensional CNN Block ---
        # 原始论文使用了7通道输入，这里保持结构，在输入端做 expand
        self.cnn_block = nn.Sequential(
            nn.Conv1d(7, 16, kernel_size=128, stride=4, padding=63), nn.BatchNorm1d(16), nn.LeakyReLU(),
            nn.Conv1d(16, 32, kernel_size=32, stride=2, padding=15), nn.BatchNorm1d(32), nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7), nn.BatchNorm1d(64), nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=8, stride=2, padding=3), nn.BatchNorm1d(64), nn.LeakyReLU(),
        )

        # --- 2. ECA Module ---
        self.eca = ECA(channels=64)

        # --- 3. Capsule Network Layers ---
        # CNN Output: (B, 64, 16)
        self.primary_caps = nn.Conv1d(64, 8 * 20, kernel_size=4, stride=2, padding=1)
        self.primary_caps_dim = 20

        # Conv1d(64, 160, k=4, s=2) on input len 16 -> Output len 8
        # Output shape: (B, 160, 8) -> Reshaped to (B, 8, 160) -> Viewed as (B, 64, 20)
        self.num_primary_caps = 8 * 8  # 64

        # Digit Capsule Layer
        self.digit_caps = MaximizedAggregationRouting(
            in_caps=self.num_primary_caps,
            out_caps=configs.model.num_classes,
            in_dim=self.primary_caps_dim,
            out_dim=configs.model.digit_caps_dim,
            num_iterations=configs.model.routing_iterations
        )

        self.optimizer = optim.Adam(self.parameters(), lr=configs.lr, weight_decay=1e-4)

        # --- Performance tracking ---
        self.best_acc = -1.0
        self.best_f1 = -1.0
        self.best_auc = -1.0
        self.best_recall = -1.0
        self.best_precision = -1.0
        self.early_stop_counter = 0

    def forward(self, x):
        # x shape: (B, 7, L)
        x = self.cnn_block(x)  # -> (B, 64, L')
        x = self.eca(x)  # -> (B, 64, L')

        # Primary Capsules
        x = self.primary_caps(x)  # -> (B, 160, 8)
        x = x.transpose(1, 2).contiguous()  # -> (B, 8, 160)
        x = x.view(x.size(0), -1, self.primary_caps_dim)  # -> (B, 64, 20)

        # Digit Capsules with Routing
        digit_caps_output = self.digit_caps(x)  # -> (B, num_classes, digit_caps_dim)
        return digit_caps_output

    def margin_loss(self, y_pred_vectors, y_true):
        m_plus = 0.9
        m_minus = 0.1
        lambda_val = 0.5

        # 计算模长作为概率: (B, num_classes)
        v_k = torch.sqrt((y_pred_vectors ** 2).sum(dim=2))  # shape (B, C)

        y_true_one_hot = F.one_hot(y_true, num_classes=self.configs.model.num_classes).float()

        loss_plus = y_true_one_hot * F.relu(m_plus - v_k).pow(2)
        loss_minus = lambda_val * (1 - y_true_one_hot) * F.relu(v_k - m_minus).pow(2)

        loss = (loss_plus + loss_minus).sum(dim=1)
        return loss.mean()

    def train_model(self, train_minibatches_iterator, test_loaders):
        self.to(self.device)

        for step in range(1, self.configs.steps + 1):
            self.train()

            try:
                source_minibatches = next(train_minibatches_iterator)
            except StopIteration:
                break

            all_xs_src, all_ys_src = [], []
            for xs_src_batch, ys_src_batch in source_minibatches:
                all_xs_src.append(xs_src_batch.to(self.device))
                all_ys_src.append(ys_src_batch.to(self.device))

            xs_src, ys_src = torch.cat(all_xs_src), torch.cat(all_ys_src)

            # 处理 4D 输入 (B, 1, H, W) -> (B, 1, L)
            if xs_src.dim() == 4:
                # 展平 H 和 W 维度
                xs_src = xs_src.view(xs_src.size(0), xs_src.size(1), -1)

            # [维度适配] (B, 1, L) -> (B, 7, L)
            if xs_src.shape[1] == 1:
                xs_src = xs_src.expand(-1, 7, -1)

            # [裁剪长度]
            if xs_src.shape[2] > self.configs.signal_len:
                xs_src = xs_src[:, :, :self.configs.signal_len]

            self.optimizer.zero_grad()
            output_vectors = self.forward(xs_src)
            loss = self.margin_loss(output_vectors, ys_src)
            loss.backward()
            self.optimizer.step()

            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:

                acc, auc, prec, recall, f1 = self.test_model(test_loaders)
                avg_acc, avg_f1, avg_auc = np.mean(acc), np.mean(f1), np.mean(auc)
                avg_recall, avg_prec = np.mean(recall), np.mean(prec)
                print(f"avgacc{avg_acc}")

                if avg_acc > self.best_acc:
                    self.best_acc = avg_acc
                    self.best_f1 = avg_f1
                    self.best_auc = avg_auc
                    self.best_recall = avg_recall
                    self.best_precision = avg_prec
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.configs.early_stopping_patience and self.configs.early_stop:
                        break
        return {
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_precision': self.best_precision,
            'best_recall': self.best_recall,
            'best_f1': self.best_f1
        }

    def test_model(self, loaders):
        self.eval()
        all_acc, all_auc, all_prec, all_recall, all_f1 = [], [], [], [], []

        num_tgt_domains = len(self.configs.datasets_tgt)
        target_loaders = loaders[:num_tgt_domains]  # 只测试目标域

        with torch.no_grad():
            for loader in target_loaders:
                if loader is None: continue
                y_pred_lst, y_prob_lst, y_true_lst = [], [], []

                for x, label_fault in loader:
                    x = x.to(self.device)

                    # 处理 4D 输入 (B, 1, H, W) -> (B, 1, L)
                    if x.dim() == 4:
                        x = x.view(x.size(0), x.size(1), -1)

                    if x.shape[1] == 1:
                        x = x.expand(-1, 7, -1)

                    if x.shape[2] > self.configs.signal_len:
                        x = x[:, :, :self.configs.signal_len]

                    output_vectors = self.forward(x)

                    # 计算模长作为概率
                    probs = torch.sqrt((output_vectors ** 2).sum(dim=2))
                    y_preds = torch.argmax(probs, dim=1)
                    y_probs_normalized = F.softmax(probs, dim=1)

                    y_pred_lst.extend(y_preds.cpu().numpy())
                    y_prob_lst.append(y_probs_normalized.cpu().numpy())
                    y_true_lst.extend(label_fault.cpu().numpy())

                if not y_true_lst: continue

                y_true, y_pred, y_prob = np.array(y_true_lst), np.array(y_pred_lst), np.vstack(y_prob_lst)
                acc, auc, prec, recall, f1 = cal_index(y_true, y_pred, y_prob)

                all_acc.append(acc)
                all_auc.append(auc)
                all_prec.append(prec)
                all_recall.append(recall)
                all_f1.append(f1)

        self.train()
        return all_acc, all_auc, all_prec, all_recall, all_f1


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(seed, configs):
    # =================================================================
    # ========== 0. 初始化 Logger =================
    # =================================================================
    # [新增] 初始化 Logger，防止 logger 未定义的错误
    dir_prefix = time.strftime("%Y%m%d%H%M%S")
    log_dir = os.path.join('logs', 'MAACCN', f'{dir_prefix}_{configs.fan_section}')
    os.makedirs(log_dir, exist_ok=True)

    set_random_seed(seed)

    # =================================================================
    # ========== 1. Determine Mode and Define Domains =================
    # =================================================================

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
    model = MAACCN(configs)

    best_results = model.train_model(
        train_minibatches_iterator, test_loaders_tgt + test_loaders_src
    )

    # --- 保存最佳结果 ---
    if best_results and best_results.get('best_acc', -1) > -1:
        save_dir = os.path.join('result', 'MAACCN')
        os.makedirs(save_dir, exist_ok=True)
        result_filename = f"section{configs.fan_section}_best_result.txt"
        result_filepath = os.path.join(save_dir, result_filename)
        file_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(result_filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{file_timestamp}] (seed: {seed})\n")
                f.write(f"Best ACC:\n{best_results['best_acc']:.4f}\n")
                f.write(f"Best AUC:\n{best_results['best_auc']:.4f}\n")
                f.write(f"Best Precision:\n{best_results['best_precision']:.4f}\n")
                f.write(f"Best Recall:\n{best_results['best_recall']:.4f}\n")
                f.write(f"Best F1:\n{best_results['best_f1']:.4f}\n")
                f.write("-" * 40 + "\n\n")
        except Exception as e:
            print(f"Failed to write results to file: {e}")


# Namespace 工具类
def to_namespace(d: dict):
    if not isinstance(d, dict):
        return d
    for key, val in d.items():
        if isinstance(val, dict):
            d[key] = to_namespace(val)
        elif isinstance(val, (list, tuple)):
            d[key] = [to_namespace(x) if isinstance(x, dict) else x for x in val]
    return types.SimpleNamespace(**d)


if __name__ == '__main__':
    with open(os.path.join(sys.path[0], 'config_files/MAACCN.yaml'), 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = to_namespace(configs)

    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
    else:
        configs.device = 'cpu'
    print(f"Using device: {configs.device}")

    # scenarios_to_test = ['00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11', 's12', 's13', 's14']
    scenarios_to_test = ['00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11',]
    run_times_per_scenario = 1

    for scenario in scenarios_to_test:
        configs.fan_section = scenario
        print(f"\n{'=' * 20} TESTING SCENARIO: {scenario} {'=' * 20}")
        for i in range(run_times_per_scenario):
            print(f"\n--- Run {i + 1}/{run_times_per_scenario} for scenario {scenario} ---")
            # 传入循环索引作为种子，确保每次运行随机性不同但可复现
            main(i, configs)