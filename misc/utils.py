import torch


def set_pytorch_precision(float_precision):
    if float_precision == 32:
        torch.set_default_dtype(torch.float32)
    elif float_precision == 64:
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f'Unsupported pytorch_precision option ({float_precision})')


def print_model_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            pass
            print(name)
            print(param.data)
    print()
    return param.data


def discrete_NAU_W_test(W):
    # checks if discretised NAU weights match the expected NAU weights (i.e. solution)
    # return 1 if success, else 0
    discrete_W = torch.where(W <= -0.5, torch.empty(W.shape).fill_(-1.),
                             torch.where(W >= 0.5, torch.ones(W.shape), torch.zeros_like(W)))
    result = 0
    if torch.all(torch.abs(discrete_W) == torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0]])):
        result = 1
    elif torch.all(torch.abs(discrete_W) == torch.Tensor([[0, 1, 0, 0], [1, 0, 0, 0]])):
        result = 1
    return result


def print_mnist_cell_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad and 'recurent_cell' in name:
            print(name)
            print(param.data)
    print()
    return param.data
