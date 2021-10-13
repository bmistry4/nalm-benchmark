import math

import torch


def huber(t, delta=1):
    # calc huber value of input 
    # delta = manually tuned
    huber_tensor = torch.zeros_like(t)
    mask = torch.abs(t) <= delta
    huber_tensor[mask] = (t[mask] ** 2 / 2.)
    huber_tensor[~mask] = (delta * (t[~mask].abs() - (0.5 * delta)))
    return huber_tensor


def mnac(x, W, mode='prod', is_training=None):
    out_size, in_size = W.size()
    x = x.view(x.size()[0], in_size, 1)
    W = W.t().view(1, in_size, out_size)

    if mode == 'prod':
        return torch.prod(x * W + 1 - W, -2)
    elif mode == 'exp-log':
        return torch.exp(torch.sum(torch.log(x * W + 1 - W), -2))
    elif mode == 'no-idendity':
        return torch.prod(x * W, -2)
    elif mode == 'div':
        # Name: identity-conversion-approx-1000-tanh1000_absApprox
        W_abs = torch.tanh(1000 * W) ** 2 if is_training else W.abs()
        smooth_zero_fire = (1 - torch.tanh(1000 * W) ** 2) if is_training else (1 - W.abs())
        return torch.prod(x.sign() * x.abs() ** W * W_abs + smooth_zero_fire, -2)
    elif mode == 'div-sepSign':
        # Name: tanh1000-SepSign -> calc magnitude and sign separately and then multiply
        W_abs = torch.tanh(1000 * W) ** 2 if is_training else W.abs()
        smooth_zero_fire = (1 - torch.tanh(1000 * W) ** 2) if is_training else (1 - W.abs())
        output = torch.prod(x.abs() ** W * W_abs + smooth_zero_fire, -2)
        output_sign = torch.prod(torch.sign(x) ** W.round(), -2)
        return output * output_sign
    else:
        raise ValueError(f'mnac mode "{mode}" is not implemented')
