
import torch
import collections
from ..writer import DummySummaryWriter

class NoRandomScope:
    def __init__(self, module):
        self._module = module

    def __enter__(self):
        self._module._disable_random()

    def __exit__(self, type, value, traceback):
        self._module._enable_random()
        return False

class ExtendedTorchModule(torch.nn.Module):
    def __init__(self, default_name, *args, writer=None, name=None, use_robustness_exp_logging=False, **kwargs):
        super().__init__()
        if writer is None:
            writer = DummySummaryWriter()

        self.writer = writer.namespace(default_name if name is None else name)
        self.allow_random = True
        self.use_robustness_exp_logging = use_robustness_exp_logging

    def set_parameter(self, name, value):
        parameter = getattr(self, name, None)
        if isinstance(parameter, torch.nn.Parameter):
            parameter.fill_(value)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.set_parameter(name, value)

    def regualizer(self, merge_in=None):
        regualizers = collections.defaultdict(int)

        if merge_in is not None:
            for key, value in merge_in.items():
                self.writer.add_scalar(f'regualizer/{key}', value)
                regualizers[key] += value

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                for key, value in module.regualizer().items():
                    regualizers[key] += value

        return regualizers

    def optimize(self, loss):
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.optimize(loss)

    def log_gradients(self):
        for name, parameter in self.named_parameters(recurse=False):
            if parameter.requires_grad:
                gradient, *_ = parameter.grad.data
                self.writer.add_summary(f'{name}/grad', gradient)
                self.writer.add_histogram(f'{name}/grad', gradient)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.log_gradients()

    def log_gradient_elems(self):
        for name, parameter in self.named_parameters(recurse=False):
            if parameter.requires_grad:
                gradient = parameter.grad.data
                self.writer.add_grad_summary(f'{name}/grad', gradient)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.log_gradient_elems()

    def log_learnable_parameters(self):
        # log each learnable weight in the network.
        # Individual elements for a parameter can be accessed via 1D indexing during the parsing stage.
        for name, parameter in self.named_parameters(recurse=False):
            if parameter.requires_grad:
                for i, elem in enumerate(parameter.view(-1)):
                    self.writer.add_scalar(f'param/{name}/{i}', elem, verbose_only=False)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.log_learnable_parameters()

    def log_gradients_mse_normalised(self, residual):
        # log the gradient of each unit without considering the scaling of the residual factor (in the partial derivatives when using a mse loss)
        for name, parameter in self.named_parameters(recurse=False):
            if parameter.requires_grad:
                gradient, *_ = parameter.grad.data
                # take mean because working with batches. Therefore this normalisation is a approximation
                gradient = gradient / residual.mean(0)   # normalise by removing effect of error residual (y-y_hat)
                self.writer.add_mse_grad_summary(f'{name}/grad', gradient)

        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module.log_gradients_mse_normalised(residual)

    def no_internal_logging(self):
        return self.writer.no_logging()

    def _disable_random(self):
        self.allow_random = False
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module._disable_random()

    def _enable_random(self):
        self.allow_random = True
        for module in self.children():
            if isinstance(module, ExtendedTorchModule):
                module._enable_random()

    def no_random(self):
        return NoRandomScope(self)
