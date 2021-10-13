import torch

from ..abstract import ExtendedTorchModule
from ..functional import mnac, Regualizer, sparsity_error


class INALULayer(ExtendedTorchModule):
    """
    Implements the iNALU (improved Neural Arithmetic Logic Unit) with independant weight matricies,
        independant gating and multiplicative sign retrieval

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features, **kwargs):
        super().__init__('inalu', **kwargs)
        self.in_features = in_features
        self.out_features = out_features

        self.eps = 1e-7
        self.omega = 20.

        self.W_a_hat = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.M_a_hat = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.W_m_hat = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.M_m_hat = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.g = torch.nn.Parameter(torch.Tensor(out_features))

        self._regualizer = Regualizer(support='inalu', type='bias', shape='linear')

    def regualizer(self):
        return super().regualizer({
            'inalu': self._regualizer([self.W_a_hat, self.M_a_hat, self.W_m_hat, self.M_m_hat, self.g]),
        })

    def reset_parameters(self):
        # see https://github.com/daschloer/inalu/blob/e41d80d3506ac0bf4a2971c262003468feca187d/nalu_architectures.py#L4
        mu_g = 0.0
        mu_m = 0.5
        mu_w = 0.88

        sd_g = 0.2
        sd_m = 0.2
        sd_w = 0.2

        torch.nn.init.normal_(self.W_a_hat, mean=mu_w, std=sd_w)
        torch.nn.init.normal_(self.M_a_hat, mean=mu_m, std=sd_m)
        torch.nn.init.normal_(self.W_m_hat, mean=mu_w, std=sd_w)
        torch.nn.init.normal_(self.M_m_hat, mean=mu_m, std=sd_m)
        torch.nn.init.normal_(self.g, mean=mu_g, std=sd_g)

    def forward(self, x):
        # [I,O]
        # weights for the summative path
        W_a = torch.tanh(self.W_a_hat) * torch.sigmoid(self.M_a_hat)
        # weights for the multiplicative m
        W_m = torch.tanh(self.W_m_hat) * torch.sigmoid(self.M_m_hat)

        self.writer.add_histogram('W_a', W_a)
        self.writer.add_tensor('W_a', W_a)
        sparsity_error_W_a = sparsity_error(W_a)
        self.writer.add_scalar('W_a/sparsity_error', sparsity_error_W_a, verbose_only=False)

        self.writer.add_histogram('W_m', W_m)
        self.writer.add_tensor('W_m', W_m)
        sparsity_error_W_m = sparsity_error(W_m)
        self.writer.add_scalar('W_m/sparsity_error', sparsity_error_W_m, verbose_only=False)
        self.writer.add_scalar('W/sparsity_error', (sparsity_error_W_a + sparsity_error_W_m) / 2., verbose_only=False)

        # [B,O] = [B,I] @ [I,O]
        a = x @ W_a
        # [B,O] = [B,I] @ [I,O]
        mz = torch.log(torch.max(torch.abs(x), torch.full(x.shape, self.eps, dtype=torch.float32))) @ W_m
        # [B,O]
        m = torch.exp(torch.min(mz, torch.full(mz.shape, self.omega, dtype=torch.float32)))
        # [B,O] = mnac([B,I], [I,O].t())
        m_sign = mnac(x.sign(), W_m.t().abs(), mode='prod')
        # [O]
        g = torch.sigmoid(self.g)
        # [B,O] = [O] * [B,O] + ([1] - [O]) * [B,O] * [B,O]
        out = g * a + (1 - g) * m * m_sign

        self.writer.add_tensor('gate', self.g)
        self.writer.add_histogram('gate', g)
        self.writer.add_scalar('gate/mean', torch.mean(g))
        self.writer.add_scalar('gate/sparsity_error', sparsity_error(g), verbose_only=False)

        self.writer.add_histogram('add', a)
        self.writer.add_histogram('mul', m)
        self.writer.add_histogram('sign', m_sign)

        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
