import math
import os

import numpy as np
import torch
import torch.utils.data
from scipy.stats import truncnorm  # must be at least version 1.6.2 (otherwise sampling implementation is too slow)

from ._dataloader import FastDataLoader


class ARITHMETIC_FUNCTIONS_STRINGIY:
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

    @staticmethod
    def reciprocal(a, *extra):
        return f'1/({a})'

class ARITHMETIC_FUNCTIONS:
    @staticmethod
    def add(*subsets):
        return np.sum(subsets, axis=0)

    @staticmethod
    def sub(a, b, *extra):
        return a - b

    @staticmethod
    def mul(*subsets):
        return np.prod(subsets, axis=0)

    def div(a, b, *extra):
        return a / b

    def squared(a, *extra):
        return a * a

    def root(a, *extra):
        return np.sqrt(a)

    @staticmethod
    def reciprocal(a, *extra):
        return 1 / a

class SimpleFunctionDataset:
    BENFORD_PERCENTAGES = [0, 0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]

    def __init__(self, operation, input_size, dist_params: tuple = tuple(['uniform']),
                 subset_ratio=0.25,
                 overlap_ratio=0.5,
                 num_subsets=2,
                 simple=False,
                 seed=None,
                 use_cuda=False,
                 max_size=2**32-1):
        super().__init__()
        self._operation_name = operation
        self._operation = getattr(ARITHMETIC_FUNCTIONS, operation)
        self._max_size = max_size
        self._use_cuda = use_cuda
        self._rng = np.random.RandomState(seed)
        self._dist_params = dist_params

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
            f'sum(v[{start}:{end}])' for start, end in self.subset_ranges
        ]
        return getattr(ARITHMETIC_FUNCTIONS_STRINGIY, self._operation_name)(*subset_str)

    def get_input_size(self):
        return self._input_size

    def fork(self, shape, sample_range, seed=None, use_extrapolation=False):
        assert shape[-1] == self._input_size
        # Added: Windows machine requires dtype to be specified as unit64
        if os.name == 'nt':
            rng = np.random.RandomState(self._rng.randint(0, 2**32 - 1, dtype='uint64') if seed is None else seed)
        else:
            rng = np.random.RandomState(self._rng.randint(0, 2**32 - 1) if seed is None else seed)

        return SimpleFunctionDatasetFork(self, shape, sample_range, rng, use_extrapolation)

class SimpleFunctionDatasetFork(torch.utils.data.Dataset):
    def __init__(self, parent, shape, sample_range, rng, use_extrapolation):
        super().__init__()

        if not isinstance(sample_range[0], list):
            sample_range = [sample_range]
        else:
            if (sample_range[0][0] - sample_range[0][1]) != (sample_range[1][0] - sample_range[1][1]):
                raise ValueError(f'unsymetric range for {sample_range}')

        self._shape = shape
        self._sample_range = sample_range
        self._rng = rng
        self.use_extrapolation = use_extrapolation

        self._operation = parent._operation
        self._input_size = parent._input_size
        self._max_size = parent._max_size
        self._use_cuda = parent._use_cuda

        self._subset_ranges = parent.subset_ranges
        self._dist_params: tuple = parent._dist_params

    def _sample(self, lower_bound, upper_bound, batch_size):
        distribution_family = self._dist_params[0]
        if distribution_family == 'uniform':
            return self._rng.uniform(
                low=lower_bound,
                high=upper_bound,
                size=(batch_size,) + self._shape)

        elif distribution_family == 'truncated-normal':
            _, in_mean, in_std, ex_mean, ex_std = self._dist_params
            if self.use_extrapolation:
                _, _, _, mean, std = self._dist_params
            else:
                _, mean, std, _, _ = self._dist_params
            lower, upper = (lower_bound - mean) / std, (upper_bound - mean) / std
            return truncnorm.rvs(lower, upper, mean, std, random_state=self._rng, size=(batch_size,) + self._shape)

        elif distribution_family == 'exponential':
            return self._rng.exponential(scale=self._dist_params[1], size=(batch_size,) + self._shape)
        elif distribution_family == 'benford':
            """
                Sample batches of integers using the Benford distribution.
                Lower bound must be at least 10, max is up to (not including) upperbound.
                Upper bounds should only be powers of 10, with the minimum upper bound being 100
                Not compatible with negative ranges.
            """
            # assert lower_bound >= 10
            # assert upper_bound == 100 or upper_bound == 1000 or upper_bound == 1000, \
            #     "Sampling from Benford's distribution only supports upper bounds 100 or 100 or 1000"
            # generate the leftmost digit which will follow the Benford's distribution
            first_digit = self._rng.choice(list(range(0, 10)), size=(batch_size,) + self._shape,
                                           p=SimpleFunctionDataset.BENFORD_PERCENTAGES)
            # generate remaining digits between lower_bound/10 to upper_bound/10
            offset_digits = self._rng.randint(lower_bound * 10**-1, upper_bound * 10 ** -1, size=(batch_size,) + self._shape)
            # calc the power to shift the first s.f. by so the offset can be added on without changing the first s.f.
            # e.g. if the offset value = 24, then the power will be 2 to raise the first s.f. by i.e., 10**2 = 100
            offset_powers = np.vectorize(lambda x: len(str(x)))(offset_digits)
            # append offset with the first digit
            samples = first_digit * 10 ** offset_powers + offset_digits
            return samples

    def _multi_sample(self, batch_size):
        if len(self._sample_range) == 1:
            return self._sample(self._sample_range[0][0], self._sample_range[0][1], batch_size)

        elif len(self._sample_range) == 2 or len(self._sample_range) == 3:
            part_0 = self._sample(self._sample_range[0][0], self._sample_range[0][1], batch_size)
            part_1 = self._sample(self._sample_range[1][0], self._sample_range[1][1], batch_size)

            if len(self._sample_range) == 2:
                choose = self._rng.randint(2, size=(batch_size,) + self._shape)
                return np.where(choose, part_0, part_1)
            # Annoyingly inelegant solution, but minimal engineering required
            # case: want each half of the input vector to be made up of a different range. E.g.[[1,2,-1,-2],[1,7,-4,-1]]
            elif len(self._sample_range) == 3:
                # update the last half of each batch item array of the part 1 with part 2.
                # If input size is odd, then the smaller half will be part 1
                part_1[:, 0:part_0.shape[1] // 2] = part_0[:, 0:part_0.shape[1] // 2]
                return part_1
            else:
                raise NotImplemented()

        else:
            raise NotImplemented()

    def __getitem__(self, select):
        # Assume select represent a batch_size by using self[0:batch_size]
        batch_size = select.stop - select.start if isinstance(select, slice) else 1

        input_vector = self._multi_sample(batch_size)

        # Compute a and b values
        sum_axies = tuple(range(1, 1 + len(self._shape)))
        subsets = [
            np.sum(input_vector[..., start:end], axis=sum_axies)
            for start, end in self._subset_ranges
        ]

        # Compute result of arithmetic operation
        output_scalar = self._operation(*subsets)[:, np.newaxis]

        # If select is an index, just return the content of one row
        if not isinstance(select, slice):
            input_vector = input_vector[0]
            output_scalar = output_scalar[0]

        return (
            torch.tensor(input_vector, dtype=torch.get_default_dtype()),
            torch.tensor(output_scalar, dtype=torch.get_default_dtype())
        )

    def __len__(self):
        return self._max_size

    def dataloader(self, batch_size=128):
        return FastDataLoader(self, batch_size, self._use_cuda)
