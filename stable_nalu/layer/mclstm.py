"""
Copied from https://github.com/hoedt/stable-nalu/blob/master/stable_nalu/layer/mclstm.py

@report{mclstm,
   author = {Hoedt, Pieter-Jan and Kratzert, Frederik and Klotz, Daniel and Halmich, Christina and Holzleitner, Markus and Nearing, Grey and Hochreiter, Sepp and Klambauer, G{\"u}nter},
   title = {MC-LSTM: Mass-Conserving LSTM},
   institution = {Institute for Machine Learning, Johannes Kepler University, Linz},
   type = {preprint},
   date = {2021},
   url = {http://arxiv.org/abs/2101.05186},
   eprinttype = {arxiv},
   eprint = {2101.05186},
}
"""

import torch
from torch import nn


class MCLSTMCell(nn.Module):

    def __init__(self, in_features: int, out_features: int,
                 cumulative: bool = False, **kwargs):
        super().__init__()
        self.mass_input_size = in_features
        self.aux_input_size = 1
        self.hidden_size = out_features
        self.normaliser = nn.Softmax(dim=-1)
        self.cumulative = cumulative

        self.out_gate = Gate(self.hidden_size, self.aux_input_size)
        self.junction = get_redistribution(kwargs.get('junct', "gate"),
                                           num_states=self.mass_input_size,
                                           num_features=self.aux_input_size,
                                           num_out=self.hidden_size,
                                           normaliser=self.normaliser)
        self.redistribution = get_redistribution(kwargs.get('redist', 'linear'),
                                                 num_states=self.hidden_size,
                                                 num_features=self.aux_input_size,
                                                 normaliser=self.normaliser)
        print(self.junction)
        print(self.redistribution)

    def reset_parameters(self):
        self.out_gate.reset_parameters()
        nn.init.constant_(self.out_gate.fc.bias, -3.)
        self.junction.reset_parameters()
        self.redistribution.reset_parameters()

    def forward(self, xt_m, xt_a, c):
        j = self.junction(xt_a)
        r = self.redistribution(xt_a)
        o = self.out_gate(xt_a)

        m_in = torch.matmul(xt_m.unsqueeze(-2), j).squeeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r).squeeze(-2)
        m_new = m_in + m_sys
        if self.cumulative:
            return m_new, m_new
        else:
            return o * m_new, (1 - o) * m_new


def get_redistribution(kind: str,
                       num_states: int,
                       num_features: int = None,
                       num_out: int = None,
                       normaliser: nn.Module = None,
                       **kwargs):
    if kind == "linear":
        return LinearRedistribution(num_states, num_features, num_out, normaliser)
    elif kind == "outer":
        return OuterRedistribution(num_states, num_features, num_out, normaliser, **kwargs)
    elif kind == "gate":
        return GateRedistribution(num_states, num_features, num_out, normaliser)
    else:
        raise ValueError("unknown kind of redistribution: {}".format(kind))


class NormalisedSigmoid(nn.Module):
    """ Normalised logistic sigmoid function. """

    def __init__(self, p: float = 1, dim: int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(s)
        return torch.nn.functional.normalize(a, p=self.p, dim=self.dim)


class Redistribution(nn.Module):
    """ Base class for modules that generate redistribution vectors/matrices. """

    def __init__(self, num_states: int, num_features: int = None, num_out: int = None, normaliser: nn.Module = None):
        """
        Parameters
        ----------
        num_states : int
            The number of states this redistribution is to be applied on.
        num_features : int, optional
            The number of features to use for configuring the redistribution.
            If the redistribution is not input-dependent, this argument will be ignored.
        num_out : int, optional
            The number of outputs to redistribute the states to.
            If nothing is specified, the redistribution matrix is assumed to be square.
        normaliser : Module, optional
            Function to use for normalising the redistribution matrix.
        """
        super().__init__()
        self.num_features = num_features
        self.num_states = num_states
        self.num_out = num_out or num_states
        self.normaliser = normaliser or NormalisedSigmoid(dim=-1)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclass must implement this method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self._compute(x)
        return self.normaliser(r)


class Gate(Redistribution):
    """
    Classic gate as used in e.g. LSTMs.

    Notes
    -----
    The vector that is computed by this module gives rise to a diagonal redistribution matrix,
    i.e. a redistribution matrix that does not really redistribute (not normalised).
    """

    def __init__(self, num_states, num_features, num_out=None, sigmoid=None):
        super().__init__(num_states, num_features, 1, sigmoid or nn.Sigmoid())
        self.fc = nn.Linear(num_features, num_states)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LinearRedistribution(Redistribution):
    """
    Redistribution by normalising a learned matrix.

    This module has an unnormalised version of the redistribution matrix as parameters
    and is normalised by applying a non-linearity (the normaliser).
    The redistribution does not depend on any of the input values,
    but is updated using the gradients to fit the data.
    """

    def __init__(self, num_states, num_features=0, num_out=None, normaliser=None):
        super(LinearRedistribution, self).__init__(num_states, 0, num_out, normaliser)
        self.r = nn.Parameter(torch.empty(self.num_states, self.num_out), requires_grad=True)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.num_states == self.num_out:
            nn.init.eye_(self.r)
            if type(self.normaliser) is NormalisedSigmoid:
                # shift and scale identity for identity-like sigmoid outputs
                torch.mul(self.r, 2, out=self.r)
                torch.sub(self.r, 1, out=self.r)
        else:
            nn.init.orthogonal_(self.r)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.r.unsqueeze(0)


class GateRedistribution(Redistribution):
    """
    Gate-like redistribution that only depends on input.

    This module directly computes all entries for the redistribution matrix
    from a linear combination of the input values and is normalised by the activation function.
    """

    def __init__(self, num_states, num_features, num_out=None, normaliser=None):
        super().__init__(num_states, num_features, num_out, normaliser)

        self.fc = nn.Linear(num_features, self.num_states * self.num_out)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        if self.num_states == self.num_out:
            # identity matrix initialisation for output
            with torch.no_grad():
                self.fc.bias[0::(self.num_out + 1)] = 3

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        return logits.view(-1, self.num_states, self.num_out)


class OuterRedistribution(Redistribution):
    """
    Redistribution by (weighted) outer product of two input-dependent vectors.

    This module computes the entries for the redistribution matrix as
    the outer product of two vectors that are linear combinations of the input values.
    There is an option to include a weight matrix parameter
    to weight each entry in the resulting matrix, which is then normalised using a non-linearity.
    The weight matrix parameter is updated through the gradients to fit the data.
    """

    def __init__(self, num_states, num_features, num_out=None, normaliser=None, weighted: bool = False):
        """
        Parameters
        ----------
        weighted : bool, optional
            Whether or not to use a weighted outer product.
        """
        super(OuterRedistribution, self).__init__(num_states, num_features, num_out, normaliser)
        self.weighted = weighted
        self.r = nn.Parameter(torch.empty(self.num_states, self.num_out), requires_grad=weighted)

        self.fc1 = nn.Linear(num_features, self.num_states)
        self.fc2 = nn.Linear(num_features, self.num_out)
        self.phi = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: account for effect normaliser
        nn.init.ones_(self.r)
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        a1 = self.phi(self.fc1(x))
        a2 = self.phi(self.fc2(x))
        outer = a1.unsqueeze(-1) * a2.unsqueeze(-2)

        if self.weighted:
            outer *= self.r

        return outer
