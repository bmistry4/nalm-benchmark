import itertools
import math
import numpy as np
import torch
import torch.utils.data
import os

from ._dataloader import FastDataLoader


class ARITHMETIC_FUNCTIONS_STRINGIY:
    # TODO: support sub, div, squared, root

    @staticmethod
    def add(*subsets):
        return ' + '.join(map(str, subsets))

    @staticmethod
    def sub(a, b, *extra):
        return f'{a} - {b}'

    @staticmethod
    def mul(*subsets):
        return ' * '.join(map(str, subsets))

    def div(a, b):
        return f'{a} / {b}'

    def squared(a, *extra):
        return f'{a}**2'

    def root(a, *extra):
        return f'sqrt({a})'


class ARITHMETIC_FUNCTIONS:
    @staticmethod
    def add(subset, axis):
        return np.sum(subset, axis=axis)

    # @staticmethod
    # def sub(a, b, *extra):
    #     return a - b

    @staticmethod
    def mul(subset, axis):
        return np.prod(subset, axis=axis)

    # def div(a, b, *extra):
    #     return a / b
    #
    # def squared(a, *extra):
    #     return a * a
    #
    # def root(a, *extra):
    #     return np.sqrt(a)


class SingleLayerDataset:
    def __init__(self, operation, input_size,
                 subset_ratio=1,
                 overlap_ratio=0,
                 num_subsets=1,
                 simple=False,
                 seed=None,
                 use_cuda=False,
                 max_size=2 ** 32 - 1):
        super().__init__()
        self._operation_name = operation
        self._operation = getattr(ARITHMETIC_FUNCTIONS, operation)
        self._max_size = max_size
        self._use_cuda = use_cuda
        self._rng = np.random.RandomState(seed)
        self._num_subsets = num_subsets

        if simple:
            self._input_size = 4

            self.subset_ranges = [(0, 4), (0, 2)]
        else:
            self._input_size = input_size
            subset_size = math.floor(subset_ratio * input_size)
            overlap_size = math.floor(overlap_ratio * subset_size)

            self.subset_ranges = []
            for subset_i in range(num_subsets):
                start = 0 if subset_i == 0 else self.subset_ranges[-1][1] - overlap_size
                end = start + subset_size
                self.subset_ranges.append((start, end))

            total_used_size = self.subset_ranges[-1][1]
            if total_used_size > input_size:
                raise ValueError('too many subsets given the subset and overlap ratios')

            offset = self._rng.randint(0, input_size - total_used_size + 1)
            self.subset_ranges = [
                (start + offset, end + offset)
                for start, end in self.subset_ranges
            ]

    def print_operation(self):
        subset_str = [
            f'{self._operation_name}(v[{start}:{end}))' for start, end in self.subset_ranges
        ]
        return ', '.join(subset_str)  # returning multiple output values

    def get_input_size(self):
        return self._input_size

    def get_output_size(self):
        # only applicable for SLTR
        return self._num_subsets

    def fork(self, shape, sample_range, seed=None):
        assert shape[-1] == self._input_size
        # Added: Windows machine requires dtype to be specified as unit64
        if os.name == 'nt':
            rng = np.random.RandomState(self._rng.randint(0, 2 ** 32 - 1, dtype='uint64') if seed is None else seed)
        else:
            rng = np.random.RandomState(self._rng.randint(0, 2 ** 32 - 1) if seed is None else seed)

        return SingleLayerDatasetFork(self, shape, sample_range, rng)


class SingleLayerDatasetFork(torch.utils.data.Dataset):
    def __init__(self, parent, shape, sample_range, rng):
        super().__init__()

        if not isinstance(sample_range[0], list):
            sample_range = [sample_range]
        else:
            if (sample_range[0][0] - sample_range[0][1]) != (sample_range[1][0] - sample_range[1][1]):
                raise ValueError(f'unsymetric range for {sample_range}')

        self._shape = shape
        self._sample_range = sample_range
        self._rng = rng

        self._operation = parent._operation
        self._input_size = parent._input_size
        self._max_size = parent._max_size
        self._use_cuda = parent._use_cuda

        self._subset_ranges = parent.subset_ranges

    def _multi_uniform_sample(self, batch_size):
        if len(self._sample_range) == 1:
            return self._rng.uniform(
                low=self._sample_range[0][0],
                high=self._sample_range[0][1],
                size=(batch_size,) + self._shape)
        elif len(self._sample_range) == 2:
            part_0 = self._rng.uniform(
                low=self._sample_range[0][0],
                high=self._sample_range[0][1],
                size=(batch_size,) + self._shape)

            part_1 = self._rng.uniform(
                low=self._sample_range[1][0],
                high=self._sample_range[1][1],
                size=(batch_size,) + self._shape)

            choose = self._rng.randint(
                2,
                size=(batch_size,) + self._shape)

            return np.where(choose, part_0, part_1)
        else:
            raise NotImplemented()

    def __getitem__(self, select):
        # Assume select represent a batch_size by using self[0:batch_size]
        batch_size = select.stop - select.start if isinstance(select, slice) else 1

        input_vector = self._multi_uniform_sample(batch_size)

        # Compute output values
        op_axes = tuple(range(1, 1 + len(self._shape)))
        # TODO: any operation, not just mul
        subsets = [
            self._operation(input_vector[..., start:end], axis=op_axes)
            for start, end in self._subset_ranges
        ]

        # reshape output vector to [B,O]
        output_vector = np.array(subsets).reshape(batch_size, len(subsets))

        # If select is an index, just return the content of one row
        if not isinstance(select, slice):
            input_vector = input_vector[0]
            output_vector = output_vector[0]

        return (
            torch.tensor(input_vector, dtype=torch.get_default_dtype()),
            torch.tensor(output_vector, dtype=torch.get_default_dtype())
        )

    def __len__(self):
        return self._max_size

    def dataloader(self, batch_size=128):
        return FastDataLoader(self, batch_size, self._use_cuda)
