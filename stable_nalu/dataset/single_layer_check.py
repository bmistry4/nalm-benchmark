# nalu-stable-exp]$ python_lfs_job.sh /home/bm4g15/nalu-stable-exp/stable_nalu/dataset/single_layer_check.py
from stable_nalu.dataset import SimpleFunctionStaticDataset
import torch
import csv

torch.set_default_dtype(torch.float64)
max_iterations = 999  # number of data items to generate: n-1


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


with open('stable_nalu/dataset/single_layer_check_float64.csv', mode='w', newline='') as csv_file:
    fieldnames = ['range', 'range type', 'operation', 'failure rate (1E-4)%', 'failure rate (1E-5)%',
                  'failure rate (1E-6)%', 'failure rate (1E-7)%', 'failure rate (1E-8)%', 'failure rate (equality)%', 'train type', 'target type']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for operation in ['add', 'sub', 'mul', 'div']:
        for r in [[-20, -10], [-2, -1], [-1.2, -1.1], [-0.2, -0.1], [-2, 2], [0.1, 0.2], [1, 2], [1.1, 1.2], [10, 20],
                  [-40, -20], [-6, -2], [-6.1, -1.2], [-2, -0.2], [[-6, -2], [2, 6]], [0.2, 2], [2, 6], [1.2, 6], [20, 40]]:
            # create dataset
            ds = init_dataset(operation)
            dataset = iter(ds.fork(sample_range=r).dataloader(batch_size=1))

            fails_equality = 0  # if not exact match to value of target tensor
            # fails_p3 = 0  # for datapoints where the precision of the operation was not within 1E-3
            fails_p4 = 0  # for datapoints where the precision of the operation was not within 1E-4
            fails_p5 = 0  # for datapoints where the precision of the operation was not within 1E-5
            fails_p6 = 0  # for datapoints where the precision of the operation was not within 1E-6
            fails_p7 = 0  # for datapoints where the precision of the operation was not within 1E-7
            fails_p8 = 0  # for datapoints where the precision of the operation was not within 1E-8

            for epoch_i, (x_train, t_train) in zip(range(max_iterations + 1), dataset):
                result_from_inputs = do_operation(x_train, operation).item()  # apply operation and get data item stored
                if result_from_inputs != t_train.item():
                    fails_equality += 1
                # check squared error
                # if (result_from_inputs - t_train.item()) ** 2 > 1e-3:
                #     fails_p3 += 1
                if (result_from_inputs - t_train.item()) ** 2 > 1e-4:
                    fails_p4 += 1
                if (result_from_inputs - t_train.item()) ** 2 > 1e-5:
                    fails_p5 += 1
                if (result_from_inputs - t_train.item()) ** 2 > 1e-6:
                    fails_p6 += 1
                if (result_from_inputs - t_train.item()) ** 2 > 1e-7:
                    fails_p7 += 1
                if (result_from_inputs - t_train.item()) ** 2 > 1e-8:
                    fails_p8 += 1

            writer.writerow({'range': f'U{r}',
                             'operation': operation,
                             # 'failure rate (1E-3)%': (fails_p3 / (max_iterations + 1)) * 100,
                             'failure rate (1E-4)%': (fails_p4 / (max_iterations + 1)) * 100,
                             'failure rate (1E-5)%': (fails_p5 / (max_iterations + 1)) * 100,
                             'failure rate (1E-6)%': (fails_p6 / (max_iterations + 1)) * 100,
                             'failure rate (1E-7)%': (fails_p7 / (max_iterations + 1)) * 100,
                             'failure rate (1E-8)%': (fails_p8 / (max_iterations + 1)) * 100,
                             'failure rate (equality)%': (fails_equality / (max_iterations + 1)) * 100,
                             'train type': f'{x_train.dtype}',
                             'target type': f'{t_train.dtype}',
                             })

            # with open('single_layer_check.txt', 'a') as out_file:
            #     out_file.write("* " * 7 + operation + " *" * 7 + "\n")
            #     out_file.write(f'Target tensor type: {x_train.dtype}\n')
            #     out_file.write(f'Range: U{r}\n')
            #     out_file.write(
            #         f"Failure rate (equality): {fails_equality} / {max_iterations+1} = {(fails_equality/(max_iterations+1))*100}%\n")
            #     out_file.write(
            #         f"Failure rate (1E-3): {fails_p3} / {max_iterations+1} = {(fails_p3/(max_iterations+1))*100}%\n")
            #     out_file.write(
            #         f"Failure rate (1E-4): {fails_p4} / {max_iterations+1} = {(fails_p4/(max_iterations+1))*100}%\n")
            #     out_file.write(
            #         f"Failure rate (1E-5): {fails_p5} / {max_iterations+1} = {(fails_p5/(max_iterations+1))*100}%\n")
            #     out_file.write(
            #         f"Failure rate (1E-6): {fails_p6} / {max_iterations+1} = {(fails_p6/(max_iterations+1))*100}%\n")
            #     out_file.write("\n")

        # print("* " * 7 + operation + " *" * 7)
        # # print(f'Dataset: {ds.print_operation()}')
        # print(f'Target tensor type: {x_train.dtype}')
        # print(f'Range: U{r}')
        # print(
        #     f"Failure rate (equality): {fails_equality} / {max_iterations+1} = {(fails_equality/(max_iterations+1))*100}%")
        # print(f"Failure rate (1E-4): {fails_p4} / {max_iterations+1} = {(fails_p4/(max_iterations+1))*100}%")
        # print(f"Failure rate (1E-5): {fails_p5} / {max_iterations+1} = {(fails_p5/(max_iterations+1))*100}%")
        # print(f"Failure rate (1E-6): {fails_p6} / {max_iterations+1} = {(fails_p6/(max_iterations+1))*100}%")
        # print(f"Failure rate (1E-7): {fails_p7} / {max_iterations+1} = {(fails_p7/(max_iterations+1))*100}%")
        # print()

print("Completed")
