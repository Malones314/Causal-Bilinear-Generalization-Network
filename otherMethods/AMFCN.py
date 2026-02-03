# AMFCN.py

#########################################################################
#
# 本脚本旨在复现论文《Adaptive Multi-Feature Fusion Convolutional Network
# for Fault Diagnosis》 (简称 AMFCN)
#
# 核心复现点:
# 1. 双流架构 (Dual-Stream Architecture):
#    - 时域分支 (Time Domain Branch): 提取原始信号的时间特征。
#    - 频域分支 (Frequency Domain Branch): 提取 FFT 变换后的频谱特征。
# 2. 自适应融合模块 (Adaptive Fusion Module):
#    - 使用注意力机制 (如 SE-Block 或 简单的可学习权重) 自适应融合两路特征。
# 3. 严格遵循 'MLFE.py' / 'DGCDN.py' 的数据加载与实验框架。
#
#########################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import yaml
import random
import types
from calIndex import cal_index
from datasets.loadData import ReadMIMII, ReadMIMIIDG, MultiInfiniteDataLoader

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


with open(os.path.join(sys.path[0], 'otherMethods/config_files/AMFCN.yaml')) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    print(configs)
    configs = to_namespace(configs)

    if configs.use_cuda and torch.cuda.is_available():
        configs.device = 'cuda'

# =========================================================================
#                          AMFCN 模型组件定义
# =========================================================================

class BasicBlock1D(nn.Module):
    """ 基础的一维卷积块: Conv -> BN -> ReLU -> MaxPool """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, pool=True):
        super(BasicBlock1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = pool
        if self.pool:
            self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        return x

class SEBlock1D(nn.Module):
    """ Squeeze-and-Excitation Block (1D版本) 用于自适应特征加权 """
    def __init__(self, channel, reduction=16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class FeatureExtractor(nn.Module):
    """ 特征提取器 (用于时域和频域分支) """
    def __init__(self, input_channel=1):
        super(FeatureExtractor, self).__init__()
        # 典型的 WDCNN 风格结构，第一层大卷积核
        self.layer1 = BasicBlock1D(input_channel, 16, kernel_size=64, stride=16, padding=24, pool=True)
        self.layer2 = BasicBlock1D(16, 32, kernel_size=3, stride=1, padding=1, pool=True)
        self.layer3 = BasicBlock1D(32, 64, kernel_size=3, stride=1, padding=1, pool=True)
        self.layer4 = BasicBlock1D(64, 64, kernel_size=3, stride=1, padding=1, pool=True)
        self.layer5 = BasicBlock1D(64, 64, kernel_size=3, stride=1, padding=1, pool=False) # 最后一层不池化以保留一定长度

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class AMFCN(nn.Module):
    """ AMFCN 主模型 """
    def __init__(self, configs):
        super(AMFCN, self).__init__()
        self.configs = configs
        self.device = torch.device(configs.device if configs.use_cuda and torch.cuda.is_available() else "cpu")

        # 1. 时域分支
        self.time_extractor = FeatureExtractor(input_channel=1)
        
        # 2. 频域分支
        self.freq_extractor = FeatureExtractor(input_channel=1)

        # 3. 自适应融合 (Adaptive Fusion)
        # 假设两个分支输出 channel 均为 64，拼接后为 128
        self.fusion_conv = nn.Conv1d(128, 128, kernel_size=1) # 融合通道
        self.attention = SEBlock1D(channel=128)               # 自适应加权

        # 4. 分类器
        self.flatten = nn.Flatten()
        # 需要计算 flatten 后的维度。假设输入长度 1024：
        # Layer1(stride16, pool2) -> 1024/32 = 32
        # Layer2(pool2) -> 16
        # Layer3(pool2) -> 8
        # Layer4(pool2) -> 4
        # Layer5(pool1) -> 4
        # Output: (B, 64, 4) -> Flatten -> 256
        # *注意*: 如果输入长度不同，这里需要动态计算或使用 AdaptiveAvgPool
        self.global_pool = nn.AdaptiveAvgPool1d(1) # 强制转换为 (B, 128, 1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, configs.num_classes)
        )

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=configs.lr, weight_decay=1e-4)

        # 记录最佳指标
        self.best_acc = -1.0
        self.best_auc = -1.0
        self.best_f1 = -1.0
        self.best_prec = -1.0
        self.best_recall = -1.0
        self.early_stop_counter = 0

    def forward(self, x_time):
        # x_time: (B, 1, L)

        # --- 1. 生成频域数据 ---
        x_freq_complex = torch.fft.rfft(x_time, dim=-1)

        # 手动计算模长，规避 torch.abs(complex) 触发的 NVRTC 报错
        # 模长 = sqrt(real^2 + imag^2)
        real = x_freq_complex.real
        imag = x_freq_complex.imag
        x_freq = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        # 后续逻辑保持不变
        max_val = torch.max(x_freq, dim=-1, keepdim=True)[0]
        x_freq = x_freq / (max_val + 1e-8)

        if x_freq.shape[-1] != x_time.shape[-1]:
            x_freq = F.interpolate(x_freq, size=x_time.shape[-1], mode='linear', align_corners=False)

        # --- 2. 特征提取 ---
        f_time = self.time_extractor(x_time) # (B, 64, L')
        f_freq = self.freq_extractor(x_freq) # (B, 64, L')

        # --- 3. 特征融合 ---
        # 拼接
        f_concat = torch.cat((f_time, f_freq), dim=1) # (B, 128, L')
        # 融合卷积
        f_fused = self.fusion_conv(f_concat)
        # 自适应加权 (Attention)
        f_att = self.attention(f_fused)
        
        # --- 4. 分类 ---
        f_global = self.global_pool(f_att) # (B, 128, 1)
        f_flat = self.flatten(f_global)    # (B, 128)
        logits = self.classifier(f_flat)
        
        return logits

    def train_model(self, train_iterator, test_loaders):
        """ 训练循环，结构参考 MLFE.py """
        self.to(self.device)

        criterion = nn.CrossEntropyLoss()

        for step in range(1, self.configs.steps + 1):
            self.train()
            
            # 获取源域数据
            try:
                source_minibatches = next(train_iterator)
            except StopIteration:
                # 重新创建迭代器（如果需要）或依靠 MultiInfiniteDataLoader 的无限特性
                continue

            all_x, all_y = [], []
            for x, y in source_minibatches:
                all_x.append(x.to(self.device))
                all_y.append(y.to(self.device))
            
            x_src = torch.cat(all_x)
            y_src = torch.cat(all_y)
            
            # 确保输入维度正确 (B, 1, L)
            if x_src.dim() == 2:
                # (B, L) -> (B, 1, L)
                x_src = x_src.unsqueeze(1)
            elif x_src.dim() == 4:
                # (B, 1, L, 1) -> (B, 1, L)
                # 使用 view 强制展平为 3D，这是最稳妥的方法
                b, c, h, w = x_src.shape
                x_src = x_src.view(b, c, -1)
            
            # 前向传播
            self.optimizer.zero_grad()
            logits = self.forward(x_src)
            loss = criterion(logits, y_src)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 验证与日志
            if step % self.configs.checkpoint_freq == 0 or step == 1 or step == self.configs.steps:

                # 测试
                acc, auc, prec, recall, f1 = self.test_model(test_loaders)
                
                # 计算平均指标
                avg_acc = np.mean(acc)
                avg_auc = np.mean(auc)
                avg_f1  = np.mean(f1)
                avg_prec = np.mean(prec)
                avg_recall = np.mean(recall)


                # 保存最佳结果
                if avg_acc > self.best_acc:
                    self.best_acc = avg_acc
                    self.best_auc = avg_auc
                    self.best_f1 = avg_f1
                    self.best_prec = avg_prec
                    self.best_recall = avg_recall
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    # 早停机制 (可选)
                    if self.configs.early_stop and self.early_stop_counter >= self.configs.early_stopping_patience:
                         break

        return {
            'best_acc': self.best_acc,
            'best_auc': self.best_auc,
            'best_precision': self.best_prec,
            'best_recall': self.best_recall,
            'best_f1': self.best_f1
        }

    def test_model(self, loaders):
        """ 测试循环，仅在目标域上评估 """
        self.eval()
        accs, aucs, precs, recalls, f1s = [], [], [], [], []
        
        # 只取目标域的 loader
        num_tgt = len(self.configs.datasets_tgt)
        target_loaders = loaders[:num_tgt]

        with torch.no_grad():
            for loader in target_loaders:
                if loader is None: continue
                y_true, y_pred, y_probs = [], [], []

                for x, y in loader:
                    x = x.to(self.device)
                    if x.dim() == 2:
                        x = x.unsqueeze(1)
                    elif x.dim() == 4:
                        b, c, h, w = x.shape
                        x = x.view(b, c, -1)
                    
                    logits = self.forward(x)
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    y_probs.append(probs.cpu().numpy())
                
                if not y_true: continue
                
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                y_probs = np.vstack(y_probs)
                
                acc, auc, prec, recall, f1 = cal_index(y_true, y_pred, y_probs)
                accs.append(acc)
                aucs.append(auc)
                precs.append(prec)
                recalls.append(recall)
                f1s.append(f1)
        
        self.train()
        return accs, aucs, precs, recalls, f1s

# =========================================================================
#                          主执行逻辑
# =========================================================================

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(seed, configs):
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

    # 初始化模型
    model = AMFCN(configs)

    all_test_loaders = test_loaders_tgt + test_loaders_src
    valid_test_loaders = [loader for loader in all_test_loaders if loader is not None]

    loss_acc_result = model.train_model(
        train_minibatches_iterator,
        valid_test_loaders,
    )

    # 结果保存逻辑保持不变
    print("===========================================================================================")
    print(f"best acc:{model.best_acc:.4f}, best auc:{model.best_auc:.4f}, best f1-score:{model.best_f1:.4f}")
    save_dir = os.path.join('result', 'CCN')
    filename = f'section0{configs.fan_section}_best_result.txt'
    file_path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    try:
        with open(file_path, 'a') as f:
            f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            f.write(f"\nBest ACC: {model.best_acc:.4f}")
            f.write(f"\nBest AUC: {model.best_auc:.4f}")
            f.write(f"\nBest precision: {model.best_prec:.4f}")  # 修正变量名
            f.write(f"\nBest recall: {model.best_recall:.4f}")
            f.write(f"\nBest F1: {model.best_f1:.4f}\n")  # 修正变量名
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