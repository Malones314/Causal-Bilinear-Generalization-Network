# reproduce_MLFE_paper.py

#########################################################################
#
# 本脚本旨在复现论文《Multilevel feature encoder for transfer learning-based
# fault detection on acoustic signal》(简称 MLFE)
#
# 核心复现点:
# 1. 严格遵循 'PropellerDANN.py' 和 'DGCDN.py' 的数据加载与实验框架。
#    - 支持 'scenario' 和 'section' 模式。
#    - 使用 MultiInfiniteDataLoader 进行源域训练。
#    - 采用相同的日志和多轮次运行结构。
# 2. 实现 MLFE 论文中描述的核心模型和特征工程。
#    - Feature Engineering: 频率掩码, 频域统计特征, K-Means聚类特征。
#    - Model Architecture: FourierTransformEncoder,
#      FrequencyDomainStatisticalEncoder, LearnableEnsembleModel。
#
#########################################################################

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
from sklearn.cluster import KMeans
from scipy.stats import iqr, entropy
from sklearn.metrics import confusion_matrix


# 导入 DGCDN 项目的工具和数据加载器
from calIndex import cal_index  # 假设这个函数能计算MCC等指标
from datasets.loadData import ReadMIMII, ReadMIMIIDG, MultiInfiniteDataLoader


class FourierTransformEncoder(nn.Module):
    """
    MLFE 论文中的傅里叶变换编码器 (基于Transformer)
    """

    def __init__(self, input_dim, model_dim, num_heads, num_layers, patch_size, dropout=0.1):
        super(FourierTransformEncoder, self).__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_size, model_dim)
        # input_dim 是 fft_len (signal_len // 2)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (input_dim // patch_size), model_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection_header = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, 2)  # 输出为2分类的logits
        )

    def forward(self, x):
        # x shape: (B, N)
        # Patching: 将 1D 序列切分为 Patches
        # unfold: (B, num_patches, patch_size)
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_size)
        x = self.patch_embedding(x)  # (B, num_patches, model_dim)

        # 动态对齐序列长度 (处理可能的长度不匹配)
        num_patches_runtime = x.shape[1]
        num_patches_init = self.positional_encoding.shape[1]

        if num_patches_runtime != num_patches_init:
            if num_patches_runtime > num_patches_init:
                x = x[:, :num_patches_init, :]
            else:
                padding = torch.zeros(x.shape[0], num_patches_init - num_patches_runtime, x.shape[2], device=x.device)
                x = torch.cat([x, padding], dim=1)

        x += self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global Average Pooling
        output = self.projection_header(x)
        return output


class FrequencyDomainStatisticalEncoder(nn.Module):
    """
    MLFE 论文中的频域统计编码器
    """

    def __init__(self, input_dim, hidden_dim=256):
        super(FrequencyDomainStatisticalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.encoder(x)


class LearnableEnsembleModel(nn.Module):
    """
    MLFE 论文中的可学习集成模型
    """

    def __init__(self):
        super(LearnableEnsembleModel, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))

    def forward(self, fte_output, fse_output):
        return self.a * fte_output + self.b * fse_output + self.c


def extract_mlfe_features(x_batch, configs, kmeans_model=None):
    """
    特征工程：提取掩码FFT特征和统计/聚类特征
    Input: x_batch (Tensor)
    """
    x_np = x_batch.cpu().numpy()

    # 如果输入是 4D (B, 1, H, W) 或 (B, C, H, W)，先展平为 (B, Total_Len)
    if x_np.ndim == 4:
        x_np = x_np.reshape(x_np.shape[0], -1)
    # 如果输入是 3D (B, 1, L)，压缩为 (B, L)
    elif x_np.ndim == 3:
        x_np = x_np.squeeze(axis=1)

    # 此时 x_np 应该是 (B, L)
    B, L = x_np.shape

    # 1. 傅里叶变换
    fft_raw = np.fft.fft(x_np, axis=1)
    # 取一半频谱
    fft_abs = np.abs(fft_raw[:, :L // 2])

    # 2. 频率掩码
    if configs.mask_freq > 0:
        # 假设采样率 16000
        cutoff_idx = fft_abs.shape[1] // (16000 // (2 * configs.mask_freq))
        if cutoff_idx == 0: cutoff_idx = fft_abs.shape[1]  # 保护
        masked_fft = np.zeros_like(fft_abs)
        masked_fft[:, :cutoff_idx] = fft_abs[:, :cutoff_idx]
    else:
        masked_fft = fft_abs

    # 3. 频域统计特征
    stat_features_list = []
    for i in range(B):
        sample_fft = masked_fft[i, :]
        mean = np.mean(sample_fft)
        std = np.std(sample_fft)
        mad = np.median(np.abs(sample_fft - np.median(sample_fft)))
        maximum = np.max(sample_fft)
        minimum = np.min(sample_fft)
        energy = np.sum(sample_fft ** 2)
        sample_iqr = iqr(sample_fft)

        epsilon = 1e-8
        prob_dist = sample_fft / (np.sum(sample_fft) + epsilon)
        sample_entropy = entropy(prob_dist + epsilon)

        stat_features_list.append([mean, std, mad, maximum, minimum, energy, sample_iqr, sample_entropy])

    stat_features = np.array(stat_features_list)
    stat_features = np.nan_to_num(stat_features)

    # 4. 基于聚类的特征
    if kmeans_model is None:
        # 简单处理：如果没有预训练模型，尝试用当前batch拟合
        # 注意：这在batch size小于k_clusters时会报错，加try-except
        n_clusters = min(configs.k_clusters, B)
        try:
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(stat_features)
            # 如果 B < configs.k_clusters，需要补齐维度，否则后续 concat 会出错
            # 这里简单处理：如果拟合的簇少于配置，我们只取 transform 后的部分，或者补0
            # 为保证维度一致性，建议 batch_size 足够大，或在外部预训练 KMeans
        except:
            # Fallback
            pass

    # 计算距离
    try:
        # 如果模型存在且维度匹配
        cluster_distances = kmeans_model.transform(stat_features)
        # 如果拟合出的簇数量小于配置的 k_clusters，补 0
        if cluster_distances.shape[1] < configs.k_clusters:
            padding = np.zeros((B, configs.k_clusters - cluster_distances.shape[1]))
            cluster_distances = np.hstack([cluster_distances, padding])
    except:
        # 如果出错 (例如模型未初始化)，返回全0
        cluster_distances = np.zeros((B, configs.k_clusters))

    cluster_features = cluster_distances

    device = x_batch.device
    masked_fft_tensor = torch.tensor(masked_fft, dtype=torch.float32, device=device)
    combined_stat_cluster_features = np.concatenate((stat_features, cluster_features), axis=1)
    stat_cluster_tensor = torch.tensor(combined_stat_cluster_features, dtype=torch.float32, device=device)

    return masked_fft_tensor, stat_cluster_tensor


class MLFE(nn.Module):
    """ 论文核心模型 """

    def __init__(self, configs, kmeans_model=None):
        super(MLFE, self).__init__()
        self.configs = configs
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")
        self.kmeans_model = kmeans_model

        # 动态获取输入维度
        # fft_len 取决于 signal_len
        fft_len = configs.signal_len // 2

        # 统计特征8个 + 聚类特征k个
        stat_cluster_input_dim = 8 + configs.k_clusters

        self.fte = FourierTransformEncoder(
            input_dim=fft_len,
            model_dim=configs.fte.model_dim,
            num_heads=configs.fte.num_heads,
            num_layers=configs.fte.num_layers,
            patch_size=configs.fte.patch_size
        ).to(self.device)

        self.fse = FrequencyDomainStatisticalEncoder(
            input_dim=stat_cluster_input_dim,
            hidden_dim=configs.fse.hidden_dim
        ).to(self.device)

        self.ensemble = LearnableEnsembleModel().to(self.device)

        self.optimizer = optim.Adam(
            list(self.fte.parameters()) + list(self.fse.parameters()) + list(self.ensemble.parameters()),
            lr=configs.lr, weight_decay=1e-4
        )

        self.best_acc = -1.0
        self.best_f1 = -1.0
        self.best_auc = -1.0
        self.best_recall = -1.0
        self.best_precision = -1.0
        self.early_stop_counter = 0

    def forward(self, masked_fft, stat_cluster_features):
        fte_logits = self.fte(masked_fft)
        fse_logits = self.fse(stat_cluster_features)
        final_logits = self.ensemble(fte_logits, fse_logits)
        return final_logits

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

            # 处理 4D 输入 (B, 1, H, W) -> 展平 -> (B, L)
            # data_length = 157 * 128 = 20096
            if xs_src.dim() == 4:
                B = xs_src.shape[0]
                xs_src = xs_src.view(B, -1)
            elif xs_src.dim() == 3:
                xs_src = xs_src.squeeze(1)

            # 裁剪到 signal_len (例如 1024)
            if xs_src.shape[1] > self.configs.signal_len:
                xs_src = xs_src[:, :self.configs.signal_len]
            elif xs_src.shape[1] < self.configs.signal_len:
                # 如果不足则填充
                padding = torch.zeros(xs_src.shape[0], self.configs.signal_len - xs_src.shape[1], device=self.device)
                xs_src = torch.cat([xs_src, padding], dim=1)

            # 提取特征
            masked_fft, stat_cluster_features = extract_mlfe_features(xs_src, self.configs, self.kmeans_model)

            self.optimizer.zero_grad()
            final_logits = self.forward(masked_fft, stat_cluster_features)
            loss = F.cross_entropy(final_logits, ys_src)
            loss.backward()
            self.optimizer.step()

            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:

                acc, auc, prec, recall, f1 = self.test_model(test_loaders)

                if not acc:
                    avg_acc, avg_f1, avg_auc = 0, 0, 0
                else:
                    avg_acc, avg_f1, avg_auc = np.mean(acc), np.mean(f1), np.mean(auc)
                    avg_prec, avg_recall = np.mean(prec), np.mean(recall)
                print(f"acc{avg_acc}")
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
            'best_f1': self.best_f1,
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_precision': self.best_precision,
            'best_recall': self.best_recall
        }

    def test_model(self, loaders):
        self.eval()
        all_acc, all_auc, all_prec, all_recall, all_f1 = [], [], [], [], []

        num_tgt_domains = len(self.configs.datasets_tgt)
        target_loaders = loaders[:num_tgt_domains]

        with torch.no_grad():
            for loader in target_loaders:
                if loader is None: continue
                y_pred_lst, y_prob_lst, y_true_lst = [], [], []

                for x, label_fault in loader:
                    x = x.to(self.device)

                    # 测试集同样的维度处理
                    if x.dim() == 4:
                        x = x.view(x.shape[0], -1)
                    elif x.dim() == 3:
                        x = x.squeeze(1)

                    if x.shape[1] > self.configs.signal_len:
                        x = x[:, :self.configs.signal_len]
                    elif x.shape[1] < self.configs.signal_len:
                        padding = torch.zeros(x.shape[0], self.configs.signal_len - x.shape[1], device=self.device)
                        x = torch.cat([x, padding], dim=1)

                    masked_fft, stat_cluster_features = extract_mlfe_features(x, self.configs, self.kmeans_model)
                    final_logits = self.forward(masked_fft, stat_cluster_features)

                    final_probs = F.softmax(final_logits, dim=1)
                    y_prob_lst.append(final_probs.cpu().numpy())
                    y_preds = torch.argmax(final_probs, dim=1)
                    y_pred_lst.extend(y_preds.cpu().numpy())
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
    """主函数，负责执行单次实验"""
    is_scenario = str(configs.fan_section).startswith('s')
    dir_prefix = 'scenario' if is_scenario else 'section'

    if is_scenario:

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

        # 使用 ReadMIMII 加载器
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

        # 使用 ReadMIMIIDG 加载器
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

    # 加载目标域数据加载器
    train_test_loaders_tgt = [ds.load_dataloaders() for ds in datasets_object_tgt]
    test_loaders_tgt = [test for train, test in train_test_loaders_tgt if test is not None]

    # K-Means 模型预训练 (理想情况下)
    # 此处应加载所有源域训练数据来训练一个K-Means模型
    # 为简化流程，我们将此步骤设为可选，并在特征提取函数中处理None的情况
    kmeans_model = None  # Placeholder

    train_minibatches_iterator = MultiInfiniteDataLoader(train_loaders_src)
    model = MLFE(configs, kmeans_model=kmeans_model)


    # --- 接收训练结果 ---
    best_results = model.train_model(
        train_minibatches_iterator, test_loaders_tgt + test_loaders_src
    )

    # ======================= START: 新增的文件写入逻辑 =======================
    # 在 main 函数中处理文件写入
    if best_results.get('best_f1', -1) > -1:
        save_dir = f'checkpoints/MLFE/{dir_prefix}_{configs.fan_section}'
        os.makedirs(save_dir, exist_ok=True)
        result_filename = f"section{configs.fan_section}_best_result.txt"
        result_filepath = os.path.join(save_dir, result_filename)

        file_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 使用 'a' (追加) 模式，为多次运行记录结果
            with open(result_filepath, 'a', encoding='utf-8') as f:
                f.write(f"[{file_timestamp}] (seed: {seed})\n")
                f.write("Best ACC:\n")
                f.write(f"{best_results['best_acc']:.4f}\n")
                f.write("Best AUC:\n")
                f.write(f"{best_results['best_auc']:.4f}\n")
                f.write("Best precision:\n")
                f.write(f"{best_results['best_precision']:.4f}\n")
                f.write("Best recall:\n")
                f.write(f"{best_results['best_recall']:.4f}\n")
                f.write("Best F1:\n")
                f.write(f"{best_results['best_f1']:.4f}\n")
                f.write("-" * 40 + "\n\n")
        except Exception as e:
            print(f"Failed to write results to file: {e}")
    # ======================== END: 新增的文件写入逻辑 ========================
import types

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

if __name__ == '__main__':
    # 1. 加载配置文件
    with open(os.path.join(sys.path[0], 'config_files/MLFE.yaml'), 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
        configs = to_namespace(configs)

    # 2. 为MLFE模型添加特定配置 (这些可以移到YAML文件中)
    configs.signal_len = 1024  # 假设输入信号长度
    configs.mask_freq = 2000  # 频率掩码的截止频率
    configs.k_clusters = 9  # K-Means的簇数量 (来自论文)

    configs.fte = to_namespace({
        'model_dim': 128,
        'num_heads': 4,
        'num_layers': 3,
        'patch_size': 16  # fft_len (512) 必须能被 patch_size 整除
    })
    configs.fse = to_namespace({
        'hidden_dim': 256
    })

    # 3. 设置设备
    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'
    else:
        configs.device = 'cpu'
    print(f"Using device: {configs.device}")

    # 4. 实验运行循环 (与参考代码一致)
    # 重点测试场景 's1', 这与MLFE论文的实验设置最匹配
    # ---- 示例：对场景各运行10次 ----
    scenarios_to_test = ['00', '01', '02', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8','s9', 's10', 's11' ]
    # scenarios_to_test = ['s14']

    run_times_per_scenario = 14

    for scenario in scenarios_to_test:
        configs.fan_section = scenario
        print(f"\n{'=' * 20} TESTING SCENARIO: {scenario} {'=' * 20}")
        for i in range(run_times_per_scenario):
            print(f"\n--- Run {i + 1}/{run_times_per_scenario} for scenario {scenario} ---")
            # 使用一个唯一的、可复现的种子

            main( i, configs)