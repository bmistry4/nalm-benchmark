import math
import torch

golden_ratio = (1 + math.sqrt(5)) / 2.
tanh = lambda x: (torch.pow(golden_ratio, 2 * x) - 1) / (torch.pow(golden_ratio, 2 * x) + 1)
sigmoid = lambda x: 1 / (1 + torch.pow(golden_ratio, -x))
