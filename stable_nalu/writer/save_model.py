
import os
import os.path as path
import torch
import numpy as np

THIS_DIR = path.dirname(path.realpath(__file__))

if 'SAVE_DIR' in os.environ:
    SAVE_DIR = os.environ['SAVE_DIR']
else:
    SAVE_DIR = path.join(THIS_DIR, '../../save')


def save_model_params(name, model):
    # saves epoch and trainable parameters of model in a txt file. Rewrites the text file for each call.
    def print_model_params(model, file=None):
        # (Duplication of method in misc/utils, but needed here to avoid circular import)
        for name, param in model.named_parameters():
            if param.requires_grad:
                pass
                print(name, file=file)
                print(param.data, file=file)
        print(file=file)
        return param.data

    save_file = path.join(SAVE_DIR, name) + '.txt'
    os.makedirs(path.dirname(save_file), exist_ok=True)
    with open(save_file, 'w') as out:
        out.write(f"epoch: {model.writer.get_iteration()}\n")
        print_model_params(model, file=out)


def save_model(name, model):
    save_file = path.join(SAVE_DIR, name) + '.pth'
    os.makedirs(path.dirname(save_file), exist_ok=True)
    torch.save(model, save_file)


def save_model_checkpoint(name, epoch, model, optimizer, rng_states):
    save_file = path.join(SAVE_DIR, name) + '.pth'
    os.makedirs(path.dirname(save_file), exist_ok=True)
    torch.save({
        'epoch': epoch,  # save epoch you want training to resume at
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'torch_rng_state': rng_states['torch'],
        'numpy_rng_state': rng_states['numpy']

    }, save_file)
    print(f'Model (/checkpoint)  trained for {epoch - 1} epochs has been saved')


def load_model(name, model, optimizer):
    load_file = path.join(SAVE_DIR, name) + '.pth'
    if not os.path.isfile(load_file):
        raise FileExistsError(f'Model save file: {load_file} (to load checkpoint) doesn\'t exist')
    checkpoint = torch.load(load_file)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['torch_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    return epoch
