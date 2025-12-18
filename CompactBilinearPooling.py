import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, projection_dim=None, signed_sqrt=True):
        """
        整合了输入投影功能的最终加强版 CompactBilinearPooling

        input_dim1: 第一个输入特征的维度
        input_dim2: 第二个输入特征的维度
        output_dim: 压缩后的高维输出维度 (例如 8192)
        projection_dim: (可选) 输入投影的目标维度。如果设置，输入将被先投影到此维度。
        signed_sqrt: 是否在最后使用 Signed Square Root 来稳定输出
        """
        super(CompactBilinearPooling, self).__init__()
        self.output_dim = output_dim
        self.signed_sqrt = signed_sqrt
        self.projection_dim = projection_dim

        # --- 整合点 1: 根据 projection_dim 决定 CBP 核心的输入维度 ---
        if self.projection_dim and self.projection_dim > 0:
            # 如果使用投影，CBP 的输入维度将是 projection_dim
            self.proj1 = nn.Linear(input_dim1, self.projection_dim)
            self.proj2 = nn.Linear(input_dim2, self.projection_dim)
            self.relu = nn.ReLU()
            cbp_input_dim1 = self.projection_dim
            cbp_input_dim2 = self.projection_dim
        else:
            # 如果不使用投影，CBP 的输入维度就是原始输入维度
            self.proj1 = None
            self.proj2 = None
            cbp_input_dim1 = input_dim1
            cbp_input_dim2 = input_dim2

        # 使用 register_buffer 注册哈希和符号张量
        # 它们的维度由 cbp_input_dim 决定
        self.register_buffer('sketch1', torch.randint(output_dim, (cbp_input_dim1,)))
        self.register_buffer('sketch2', torch.randint(output_dim, (cbp_input_dim2,)))
        self.register_buffer('sign1', torch.randint(2, (cbp_input_dim1,)) * 2 - 1)
        self.register_buffer('sign2', torch.randint(2, (cbp_input_dim2,)) * 2 - 1)

    def forward(self, x1, x2):
        """
        x1: (batch, input_dim1)
        x2: (batch, input_dim2)
        return: (batch, output_dim)
        """
        # --- 整合点 2: 如果定义了投影层，则先执行投影 ---
        if self.proj1 is not None:
            x1 = self.relu(self.proj1(x1))
            x2 = self.relu(self.proj2(x2))

        batch_size = x1.size(0)

        # 向量化的 Tensor Sketch 实现 (使用 scatter_add_)
        x1_signed = x1 * self.sign1
        x2_signed = x2 * self.sign2

        sketch1_flat = torch.zeros(batch_size, self.output_dim, device=x1.device)
        sketch2_flat = torch.zeros(batch_size, self.output_dim, device=x2.device)

        sketch_indices1 = self.sketch1.unsqueeze(0).repeat(batch_size, 1)
        sketch_indices2 = self.sketch2.unsqueeze(0).repeat(batch_size, 1)

        sketch1_flat.scatter_add_(1, sketch_indices1, x1_signed)
        sketch2_flat.scatter_add_(1, sketch_indices2, x2_signed)

        # FFT -> 点乘 -> IFFT
        fft1 = torch.fft.fft(sketch1_flat, dim=1)
        fft2 = torch.fft.fft(sketch2_flat, dim=1)
        cbp_fft = fft1 * fft2
        cbp = torch.fft.ifft(cbp_fft, dim=1).real

        if self.signed_sqrt:
            cbp = torch.sign(cbp) * torch.sqrt(torch.abs(cbp) + 1e-9)

        cbp = F.normalize(cbp, p=2, dim=1)

        return cbp