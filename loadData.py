import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler, RandomSampler, BatchSampler
import os
import scipy.io as sio
import numpy as np
from collections import Counter


# ================================== [ From DatasetClass.py ] ==================================

class _InfiniteSampler(Sampler):
    """
    无限采样器：将普通采样器包装为无限循环的采样器
    """

    def __init__(self, sampler):
        super().__init__(sampler)
        self.sampler = sampler
        self.epoch_counter = 0

    def __iter__(self):
        while True:
            self.epoch_counter += 1
            yield from iter(self.sampler)


class InfiniteDataLoader:
    """
    无限数据加载器：不断产生数据批次
    """

    def __init__(self, dataset, batch_size=128, weights=None, num_workers=0):
        super().__init__()
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a torch.utils.data.Dataset instance")

        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights,
                num_samples=int(1e10),
                replacement=True
            )
        else:
            replacement = len(dataset) < 1000
            sampler = RandomSampler(dataset, replacement=replacement)

        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False
        )

        self.dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            pin_memory=True,
            worker_init_fn=lambda _: np.random.seed()
        )
        self._infinite_iterator = iter(self.dataloader)

    def __iter__(self):
        while True:
            try:
                yield next(self._infinite_iterator)
            except StopIteration:
                self._infinite_iterator = iter(self.dataloader)
                yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError('This is an infinite dataloader!')

    def __del__(self):
        if hasattr(self, '_infinite_iterator'):
            del self._infinite_iterator


class SimpleDataset(Dataset):
    def __init__(self, data_content):
        # MODIFIED: 数据现在期望是 (N, 1, 157, 128) 的4D张量
        self.data = torch.as_tensor(data_content['data'], dtype=torch.float32)
        self.class_label = torch.as_tensor(data_content['label'], dtype=torch.long)

        if len(self.data) == 0:
            raise ValueError("数据集不能为空!")
        if torch.isnan(self.data).any():
            raise ValueError("数据包含 NaN 值!")
        if torch.isinf(self.data).any():
            raise ValueError("数据包含 Inf 值!")

        # MODIFIED: 维度检查从3D改为4D
        if self.data.ndim != 4:
            raise ValueError(f"数据维度不正确，期望4D张量 (N, C, H, W)，但得到 {self.data.ndim}D")

        if len(self.data) != len(self.class_label):
            raise ValueError(
                f"数据/标签数量不匹配: {len(self.data)} vs {len(self.class_label)}"
            )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if torch.isnan(data).any() or torch.isinf(data).any():
            raise ValueError(f"Invalid data at index {index}")
        return data, self.class_label[index]


class MultiInfiniteDataLoader:
    def __init__(self, loaders, max_retries=3):
        self.loaders = loaders
        self.iters = [iter(dl) for dl in loaders]
        self.max_retries = max_retries

        for i, loader in enumerate(loaders):
            try:
                sample = next(iter(loader))
            except StopIteration:
                raise ValueError(f"DataLoader {i} 初始化失败：无法获取任何数据")
            except Exception as e:
                raise RuntimeError(f"DataLoader {i} 初始化异常：{str(e)}")

    def __iter__(self):
        return self

    def __next__(self):
        batches = []
        for i in range(len(self.loaders)):
            retry_count = 0
            while True:
                try:
                    batch = next(self.iters[i])
                    batches.append(batch)
                    break
                except StopIteration:
                    self.iters[i] = iter(self.loaders[i])
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        raise RuntimeError(
                            f"DataLoader {i} 在 {self.max_retries} 次重试后仍无法生成数据\n"
                            f"可能原因：1.数据集为空 2.batch_size设置错误 3.数据预处理异常"
                        )
        return batches


# ================================== [ From loadData.py ] ==================================

try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = os.getcwd()


class ReadMIMIIDG():
    """
    修改后的 ReadMIMIIDG，逻辑与 ReadMIMII 保持一致：
    1. 移除全局统计量缓存和 Strict DG 归一化。
    2. 采用单域内部归一化 (Local Normalization)。
    3. 保留 Section 参数以定位 attributes_{section}_train.mat 文件。
    """

    def __init__(self, domain, seed, section, configs):
        self.configs = configs
        self.section = str(section).zfill(2)
        self.domain = domain
        self.seed = seed
        self.class_weights = None
        self.batch_size = configs.batch_size
        self.device = configs.device

    def read_data_file(self):
        # 1. 构建文件路径 (保持原有DG数据集的文件命名结构)
        data_base_path = os.path.join(PROJECT_ROOT, "Data", "fan")
        train_file = os.path.join(data_base_path, f"attributes_{self.section}_train.mat")
        test_file = os.path.join(data_base_path, f"attributes_{self.section}_test.mat")

        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"数据文件缺失: {train_file} 或 {test_file}")

        train_data_mat = sio.loadmat(train_file)
        test_data_mat = sio.loadmat(test_file)

        train_dict = {'data': torch.empty(0), 'label': torch.empty(0, dtype=torch.long)}
        test_dict = {'data': torch.empty(0), 'label': torch.empty(0, dtype=torch.long)}

        # 定义处理函数 (完全复用 ReadMIMII 的逻辑)
        def process_data(raw_data, labels):
            if raw_data.size == 0:
                return torch.empty(0), torch.empty(0, dtype=torch.long)

            # 1. 过滤 NaN
            valid_indices = ~np.isnan(raw_data).any(axis=(1, 2))
            raw_data = raw_data[valid_indices]
            labels = labels[valid_indices]

            # 对齐检查 (MIMIIDG原始数据有时候label和data长度不一致，增加此保护)
            if raw_data.shape[0] != labels.shape[0]:
                min_len = min(raw_data.shape[0], labels.shape[0])
                raw_data = raw_data[:min_len]
                labels = labels[:min_len]

            if raw_data.shape[0] == 0:
                return torch.empty(0), torch.empty(0, dtype=torch.long)

            # 2. 局部归一化 (Local Normalization) - 与 ReadMIMII 一致
            mean = np.mean(raw_data)
            std = np.std(raw_data)
            processed_data = (raw_data - mean) / (std + 1e-8)

            # 3. 转 Tensor 并增加通道维度 (N, 1, 157, 128)
            tensor_data = torch.from_numpy(processed_data).float()
            tensor_data = tensor_data.unsqueeze(1)
            tensor_labels = torch.from_numpy(labels).long()

            return tensor_data, tensor_labels

        # ==================== 1. 处理训练集 ====================
        if self.domain in train_data_mat:
            raw_data = train_data_mat[self.domain]['data'][0, 0]
            labels = train_data_mat[self.domain]['label'][0, 0].squeeze()

            train_data_tensor, train_labels_tensor = process_data(raw_data, labels)

            if len(train_data_tensor) > 0:
                train_dict['data'] = train_data_tensor
                train_dict['label'] = train_labels_tensor

                # 计算类别权重 (保持原 ReadMIMII 逻辑)
                counter = Counter(train_labels_tensor.tolist())
                total = sum(counter.values())
                sorted_classes = sorted(counter.keys())
                # 使用 MIMII 的权重计算方式: 1 - p(x)
                weights = [1.0 - (counter[cls] / total) for cls in sorted_classes]
                self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # ==================== 2. 处理测试集 ====================
        if self.domain in test_data_mat:
            raw_data = test_data_mat[self.domain]['data'][0, 0]
            labels = test_data_mat[self.domain]['label'][0, 0].squeeze()

            test_data_tensor, test_labels_tensor = process_data(raw_data, labels)

            if len(test_data_tensor) > 0:
                test_dict['data'] = test_data_tensor
                test_dict['label'] = test_labels_tensor
        else:
            # 只有当既不在训练集也不在测试集时才警告，但这里是针对测试集部分的检查
            pass

        return {'train': train_dict, 'test': test_dict}

    def load_dataloaders(self):
        # 保持原有的 loader 构建逻辑，这部分本来就差不多
        g = torch.Generator()
        g.manual_seed(self.seed)
        data_dicts = self.read_data_file()
        train_dict = data_dicts['train']
        test_dict = data_dicts['test']

        train_loader, test_loader = None, None

        # Train Loader
        if len(train_dict['data']) > 0:
            dataset_train = SimpleDataset(train_dict)
            safe_batch_size = min(self.batch_size, len(dataset_train))

            # 检查 drop_last
            should_drop_last = True
            if len(dataset_train) < self.batch_size:
                # 如果数据不够一个batch，通常 ReadMIMII 会警告或根据配置处理
                # 这里为了稳健，如果不够则不 drop_last，或者保持原配置
                pass

            train_loader = DataLoader(
                dataset_train,
                batch_size=safe_batch_size,
                shuffle=True,
                generator=g,
                num_workers=0,
                drop_last=True  # 通常训练集保持 drop_last=True
            )

        # Test Loader
        if len(test_dict['data']) > 0:
            dataset_test = SimpleDataset(test_dict)
            # 测试集 batch_size 不需要 drop_last
            safe_batch_size = min(self.batch_size, len(dataset_test))

            test_loader = DataLoader(
                dataset_test,
                batch_size=safe_batch_size,
                shuffle=False,
                generator=g,
                num_workers=0,
                drop_last=False
            )

        return train_loader, test_loader

class ReadMIMII():
    """
    读取按场景(scenario)划分的数据集的自定义类
    """

    def __init__(self, scenario, domain_id, seed, configs):
        self.configs = configs
        self.scenario = scenario
        self.domain_id = domain_id
        self.seed = seed
        self.batch_size = configs.batch_size
        self.class_weights = None

    def read_data_file(self):
        mat_file_root = os.path.join(PROJECT_ROOT, "Data", "0_dB_fan", "fan", "mat_files_scenarios")
        train_file = os.path.join(mat_file_root, f"{self.scenario}_train.mat")
        test_file = os.path.join(mat_file_root, f"{self.scenario}_test.mat")

        train_data_mat = sio.loadmat(train_file)
        test_data_mat = sio.loadmat(test_file)

        is_source_domain = self.domain_id in train_data_mat
        is_target_domain = self.domain_id in test_data_mat

        if not is_source_domain and not is_target_domain:
            raise ValueError(f"域 {self.domain_id} 在场景 {self.scenario} 的训练集和测试集中都未找到。")

        train_dict = {'data': torch.empty(0), 'label': torch.empty(0, dtype=torch.long)}
        test_dict = {'data': torch.empty(0), 'label': torch.empty(0, dtype=torch.long)}

        def process_data(raw_data, labels):
            if raw_data.size == 0:
                return torch.empty(0), torch.empty(0, dtype=torch.long)

            valid_indices = ~np.isnan(raw_data).any(axis=(1, 2))
            raw_data = raw_data[valid_indices]
            labels = labels[valid_indices]

            mean = np.mean(raw_data)
            std = np.std(raw_data)
            processed_data = (raw_data - mean) / (std + 1e-8)

            tensor_data = torch.from_numpy(processed_data).float()
            # MODIFIED: 取消展平，直接添加通道维度，以保留2D结构
            # 原始形状: (N, 157, 128) -> 修改后形状: (N, 1, 157, 128)
            tensor_data = tensor_data.unsqueeze(1)
            tensor_labels = torch.from_numpy(labels).long()
            return tensor_data, tensor_labels

        if is_source_domain:
            domain_data_train = train_data_mat[self.domain_id]
            raw_train_data = domain_data_train['data'][0, 0]
            train_labels = domain_data_train['label'][0, 0].squeeze()

            train_data_tensor, train_labels_tensor = process_data(raw_train_data, train_labels)
            train_dict['data'] = train_data_tensor
            train_dict['label'] = train_labels_tensor

            if len(train_labels_tensor) > 0:
                counter = Counter(train_labels_tensor.tolist())
                total = sum(counter.values())
                sorted_classes = sorted(counter.keys())
                weights = [1.0 - (counter[cls] / total) for cls in sorted_classes]
                self.class_weights = torch.tensor(weights, dtype=torch.float32).to(self.configs.device)

        if is_target_domain:
            domain_data_test = test_data_mat[self.domain_id]
            raw_test_data = domain_data_test['data'][0, 0]
            test_labels = domain_data_test['label'][0, 0].squeeze()

            test_data_tensor, test_labels_tensor = process_data(raw_test_data, test_labels)
            test_dict['data'] = test_data_tensor
            test_dict['label'] = test_labels_tensor

        return {'train': train_dict, 'test': test_dict}

    def load_dataloaders(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        the_data = self.read_data_file()
        train_dict = the_data['train']
        test_dict = the_data['test']

        train_loader, test_loader = None, None

        if len(train_dict['data']) > 0:
            dataset_train = SimpleDataset(train_dict)
            safe_batch_size_train = min(self.batch_size, len(dataset_train))
            if len(dataset_train) < safe_batch_size_train and getattr(self.configs, 'drop_last_train', True):
                print(
                    f"[Warning] Domain {self.domain_id} 训练集样本数({len(dataset_train)})不足一个batch，将返回空加载器。")
            else:
                train_loader = DataLoader(
                    dataset_train,
                    batch_size=safe_batch_size_train,
                    shuffle=True,
                    generator=g,
                    num_workers=0,
                    drop_last=getattr(self.configs, 'drop_last_train', True)
                )

        if len(test_dict['data']) > 0:
            dataset_test = SimpleDataset(test_dict)
            safe_batch_size_test = min(self.batch_size, len(dataset_test))
            test_loader = DataLoader(
                dataset_test,
                batch_size=safe_batch_size_test,
                shuffle=True,
                generator=g,
                num_workers=0,
                drop_last=False
            )
        return train_loader, test_loader