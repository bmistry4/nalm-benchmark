import numpy as np
from ._single_layer_abstract import SingleLayerDataset

"""
To be used in the single layer exps when you want to do experiments with multiple output neurons (rather than a scalar)
To match the current setup of the dataset used in the paper then use flags: --overlap-ratio 0 --subset-ratio 1 --num-subsets 1 --output-size 1
    - this will give the same results as the default Dataset used for the single layer exp
When using this dataset, the output-size == num-subsets !!!
    - each subset in num-subsets will be a slice of the input vector to apply an operation over
    
NOTE: atm only supports mul or add operations
"""
class SingleLayerStaticDataset(SingleLayerDataset):
    def __init__(self, operation,
                 input_size=2,
                 **kwargs):
        super().__init__(operation, input_size,
                         **kwargs)

    def fork(self, sample_range=[1, 2], *args, **kwargs):
        return super().fork((self._input_size,), sample_range, *args, **kwargs)
