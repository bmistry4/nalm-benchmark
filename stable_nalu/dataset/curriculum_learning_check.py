# nalu-stable-exp]$ python_lfs_job.sh /home/bm4g15/nalu-stable-exp/stable_nalu/dataset/curriculum_learning.py
from stable_nalu.dataset import SimpleFunctionStaticDataset
import torch
import csv

torch.set_default_dtype(torch.float32)
max_iterations = 20  # number of data items to generate: n-1


def init_dataset(op):
    dataset = SimpleFunctionStaticDataset(
        operation=op,
        input_size=2,
        subset_ratio=0.5,
        overlap_ratio=0,
        num_subsets=2,
        simple=False,
        use_cuda=False,
        seed=0,
    )
    return dataset


def do_operation(tensor, op):
    if op == 'add':
        return tensor.squeeze()[0] + tensor.squeeze()[1]
    elif op == 'sub':
        return tensor.squeeze()[0] - tensor.squeeze()[1]
    elif op == 'mul':
        return tensor.squeeze()[0] * tensor.squeeze()[1]
    elif op == 'div':
        return tensor.squeeze()[0] / tensor.squeeze()[1]
    # elif op == 'squared':
    #     return tensor.squeeze()[0] ** 2
    # elif op == 'root':
    #     return torch.sqrt(tensor.squeeze()[0])

ds = init_dataset('sub')
ds1 = iter(ds.fork(sample_range=[1.1, 1.2]).dataloader(batch_size=1))
ds2 = iter(ds.fork(sample_range=[5,10]).dataloader(batch_size=1))
ds3 = iter(ds.fork(sample_range=[15,20]).dataloader(batch_size=1))

dataset = [ds1, ds2, ds3]


# create dataset
epoch_i = 0
idx = 0
while epoch_i < max_iterations+1:
    (x_train, t_train) = next(dataset[idx])
    print(f'epoch {epoch_i}, ds: {idx}, x_train {x_train}, t_train {t_train}')
    if epoch_i % 5 == 0 and idx < len(dataset)-1 and epoch_i != 0:
        idx += 1
        print('ds change')
    epoch_i += 1
print("Completed")
