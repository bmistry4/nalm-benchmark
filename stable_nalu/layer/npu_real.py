import math

import torch

from ..abstract import ExtendedTorchModule
from ..functional import Regualizer, sparsity_error

"""
Combine with NAC to do all basic operations
2 layer -> NAU; RealNPU

--layer-type ReRegualizedLinearNAC --nac-mul real_npu --regualizer 1 --regualizer-scaling npu
"""


class RealNPULayer(ExtendedTorchModule):
    """
        Implements the Real Neural Power Unit

        Arguments:
            in_features: number of ingoing features
            out_features: number of outgoing features
        """

    def __init__(self, in_features, out_features, npu_clip='none',
                 **kwargs):
        super().__init__('real_npu', **kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.eps = torch.finfo(torch.float).eps  # 32-bit eps

        self.W_real = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.g = torch.nn.Parameter(torch.Tensor(in_features))
        self.npu_clip = npu_clip
        self.Wr_init_mode = kwargs['npu_Wr_init']

        if kwargs['regualizer_npu_w']:
            # penalise {-1,1}
            if kwargs['realnpu_reg_type'] == 'W':
                self._regualizer_W = Regualizer(
                    support='npu', type=kwargs['realnpu_reg_type'],
                    shape='linear'
                )
            # penalise {-1,0,1}
            elif kwargs['realnpu_reg_type'] == 'bias':
                self._regualizer_W = Regualizer(
                    support='realnpu', type=kwargs['realnpu_reg_type'],
                    shape='linear'
                )
        else:
            self._regualizer_W = Regualizer(zero=True)

        if kwargs['regualizer_gate']:
            self._regualizer_g = Regualizer(
                support='mnac', type='bias',
                shape='linear'
            )
        else:
            self._regualizer_g = Regualizer(zero=True)

    def reset_parameters(self):
        if self.Wr_init_mode == 'xavier-uniform':
            torch.nn.init.xavier_uniform_(self.W_real)
        elif self.Wr_init_mode == 'xavier-uniform-constrained':
            std = math.sqrt(2.0 / (self.in_features + self.out_features))
            r = min(0.5, math.sqrt(3.0) * std)
            torch.nn.init.uniform_(self.W_real, -r, r)

        torch.nn.init.ones_(self.g)
        self.g.data /= 2.0

    def optimize(self, loss):
        if self.npu_clip == 'none':
            pass
        elif self.npu_clip == 'w':
            self.W_real.data.clamp_(-1.0, 1.0)
        elif self.npu_clip == 'g':
            self.g.data.clamp_(0.0, 1.0)
        elif self.npu_clip == 'wg':
            self.W_real.data.clamp_(-1.0, 1.0)
            self.g.data.clamp_(0.0, 1.0)

    def regualizer(self):
        return super().regualizer({
            'g-NPU': self._regualizer_g(self.g),
            'W-NPU': self._regualizer_W([self.W_real])
        })

    def forward(self, x):
        self.writer.add_histogram('W_real', self.W_real)
        self.writer.add_tensor('W_real', self.W_real, verbose_only=False if self.use_robustness_exp_logging else True)
        self.writer.add_scalar('W_real/sparsity_error', sparsity_error(self.W_real), verbose_only=self.use_robustness_exp_logging)

        g_hat = torch.clamp(self.g, 0.0, 1.0)  # [in]
        self.writer.add_histogram('gate', g_hat)
        self.writer.add_tensor('gate', g_hat, verbose_only=False if self.use_robustness_exp_logging else True)
        self.writer.add_scalar('gate/sparsity_error', sparsity_error(g_hat), verbose_only=self.use_robustness_exp_logging)

        r = torch.abs(x) + self.eps                                     # [B, in]
        # * = broadcasted element-wise product
        r = g_hat * r + (1 - g_hat)                                     # [B,in] = [in] * [B, in] + (1 - [in])
        k = torch.max(-torch.sign(x), torch.zeros_like(x)) * math.pi    # [B,in] = max([B, in], [B, in]) * pi
        k = g_hat * k                                                   # [in] * [B, in]

        # [B, out] = exp([B,in][in, out])) * cos([B,in][in, out])
        z = torch.exp(torch.log(r).matmul(self.W_real)) * torch.cos(k.matmul(self.W_real))

        return z  # [B, out]

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
