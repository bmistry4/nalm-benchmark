"""
Copied from https://github.com/hoedt/stable-nalu/blob/master/stable_nalu/layer/mcfc.py.
Slight modifications to MulMCFC and addition of 2 new subclasses of MulMCFC named: MulMCFCSignINALU and
MulMCFCSignRealNPU

@report{mclstm,
   author = {Hoedt, Pieter-Jan and Kratzert, Frederik and Klotz, Daniel and Halmich, Christina and Holzleitner,
   Markus and Nearing, Grey and Hochreiter, Sepp and Klambauer, G{\"u}nter},
   title = {MC-LSTM: Mass-Conserving LSTM},
   institution = {Institute for Machine Learning, Johannes Kepler University, Linz},
   type = {preprint},
   date = {2021},
   url = {http://arxiv.org/abs/2101.05186},
   eprinttype = {arxiv},
   eprint = {2101.05186},
}
"""
import math

import torch
from torch import nn

from stable_nalu.abstract import ExtendedTorchModule
from stable_nalu.layer.mclstm import get_redistribution, Gate
from ..functional import mnac, Regualizer
from ..functional import sparsity_error


class MCFullyConnected(ExtendedTorchModule):

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__('MCFC', **kwargs)
        self.mass_input_size = in_features
        self.aux_input_size = 1
        self.hidden_size = out_features
        self.normaliser = nn.Softmax(dim=-1)

        self.out_gate = Gate(self.hidden_size, self.aux_input_size)
        self.junction = get_redistribution("linear",
                                           num_states=self.mass_input_size,
                                           num_features=self.aux_input_size,
                                           num_out=self.hidden_size,
                                           normaliser=self.normaliser)

    @torch.no_grad()
    def reset_parameters(self):
        self.out_gate.reset_parameters()
        self.junction.reset_parameters()

    def log_gradients(self):
        for name, parameter in self.named_parameters():
            gradient, *_ = parameter.grad.data
            self.writer.add_summary(f'{name}/grad', gradient)
            self.writer.add_histogram(f'{name}/grad', gradient)

    def regualizer(self, merge_in=None):
        r1 = -torch.mean(self.junction.r ** 2)
        r2 = -torch.mean(self.out_gate.fc.weight ** 2)
        r3 = -torch.mean(self.out_gate.fc.bias ** 2)
        return super().regualizer({
            'W': r1 + r2 + r3
        })

    def forward(self, x):
        x_m, x_a = x, x.new_ones(1)
        j = self.junction(x_a)
        o = self.out_gate(x_a)

        # don't really need this for the 1 layer 2-input task (since the junction should always be 1 so the SE is always 0)
        # but included for completion
        self.writer.add_tensor('mcfc_junction', j, verbose_only=False if self.use_robustness_exp_logging else True)
        self.writer.add_scalar('mcfc_junction/sparsity_error', sparsity_error(j),
                               verbose_only=self.use_robustness_exp_logging)
        self.writer.add_tensor('mcfc_out_gate', o, verbose_only=False if self.use_robustness_exp_logging else True)
        self.writer.add_scalar('mcfc_out_gate/sparsity_error', sparsity_error(o), verbose_only=self.use_robustness_exp_logging)

        m_in = torch.matmul(x_m.unsqueeze(-2), j).squeeze(-2)
        return o * m_in


class MulMCFC(ExtendedTorchModule):
    """
    Slight modification from original MulMCFC.
    When passing the out features for mcfc do not have +1 (which I presume was to act as a trash cell).
    Therefore, the return tensor in the forward does not need to index the final element.
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('MulMCFC', **kwargs)
        self.mcfc = MCFullyConnected(in_features, out_features, **kwargs)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def reset_parameters(self):
        self.mcfc.reset_parameters()
        nn.init.zeros_(self.bias)

    def _sign_fn(self, x):
        # default value
        return 1

    def forward(self, x, mod_inp_x=None):
        """
        Args:
            x: original input
            mod_inp_x: modified input. A result of child classes.

        Returns: output tensor

        """
        self.writer.add_tensor('mulmcfc_bias', self.bias,
                               verbose_only=False if self.use_robustness_exp_logging else True)
        self.writer.add_scalar('mulmcfc_bias/sparsity_error', sparsity_error(self.bias),
                               verbose_only=self.use_robustness_exp_logging)

        # update input to module with modified input or just use the original input
        x_in = mod_inp_x if mod_inp_x is not None else x
        x_sign = self._sign_fn(x)
        log_sum = self.mcfc(torch.log(x_in))             # [B,O]
        return torch.exp(log_sum + self.bias) * x_sign   # [B,O] = exp([B,O] + [O]) * [B,O]


class MulMCFCSignINALU(MulMCFC):
    """
    Subclass of MulMCFC.
    Can process negative numbers by calling the MulMCFC on |x| and reapplying the expected sign of the output.
    The sign reapplication uses the same method as the iNALU
    (https://www.frontiersin.org/articles/10.3389/frai.2020.00071/full).
    """

    def __init__(self, in_features, out_features, **kwargs):
        super(MulMCFCSignINALU, self).__init__(in_features, out_features, **kwargs)

    def _sign_fn(self, x):
        sign_weights = self.mcfc.normaliser(self.mcfc.junction.r)           # [I, O] = [I,O]
        sign = mnac(x.sign(), sign_weights.t().abs(), mode='prod')          # [B,O] = mnac([B,O], [O,I])
        return sign

    def forward(self, x):
        abs_x = torch.abs(x)
        # the mulmcfc will be passed unsigned x but the sign calculation requires the original x with signs
        return super().forward(x, mod_inp_x=abs_x)


class MulMCFCSignRealNPU(MulMCFC):
    """
    Subclass of MulMCFC.
    Can process negative numbers by calling the MulMCFC on |x| and reapplying the expected sign of the output.
    The sign reapplication uses the same method as the RealNPU (https://arxiv.org/abs/2006.01681) which requires
    learning an additional gating vector g.
    """

    def __init__(self, in_features, out_features, **kwargs):
        super(MulMCFCSignRealNPU, self).__init__(in_features, out_features, **kwargs)
        self.g = torch.nn.Parameter(torch.Tensor(in_features))

        if kwargs['regualizer_gate']:
            self._regualizer_g = Regualizer(
                support='mnac', type='bias',
                shape='linear'
            )
        else:
            self._regualizer_g = Regualizer(zero=True)

    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.ones_(self.g)
        self.g.data /= 2.0

    def regualizer(self):
        # penalise g for not being 0 or 1
        return super().regualizer({
            'g-NPU': self._regualizer_g(self.g),
        })

    def _sign_fn(self, x):
        sign_weights = self.mcfc.normaliser(self.mcfc.junction.r)       # [I, O] = [I,O]
        g_hat = torch.clamp(self.g, 0.0, 1.0)                           # [in]
        self.writer.add_tensor('mulmcfcsignrealnpu_g_hat', g_hat,
                               verbose_only=False if self.use_robustness_exp_logging else True)
        self.writer.add_scalar('mulmcfcsignrealnpu_g_hat/sparsity_error', sparsity_error(g_hat),
                               verbose_only=self.use_robustness_exp_logging)

        k = torch.max(-torch.sign(x), torch.zeros_like(x)) * math.pi    # [B,I] = max([B, I], [B, I]) * pi
        k = g_hat * k                                                   # [I] * [B, I]
        sign = torch.cos(k.matmul(sign_weights))                        # [B,O] = [B, I] * [I,O]
        return sign

    def forward(self, x):
        abs_x = torch.abs(x)
        # the mulmcfc will be passed unsigned x but the sign calculation requires the original x with signs
        return super().forward(x, mod_inp_x=abs_x)
