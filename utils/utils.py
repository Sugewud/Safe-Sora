import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist
import math
import os

def adjust_learning_rate(optimizer, sample_num, batch_idx, epoch, total_epoch, blr, start_step, min_lr=1e-6):
    """Decay the learning rate with half-cycle cosine after warmup"""
    total_step = total_epoch * sample_num
    current_step = (epoch - 1) * sample_num + batch_idx
    current_step = min(current_step + start_step, total_step)
    warmup_steps = int(total_step * 0.2)
    if current_step < warmup_steps:
        lr = blr * current_step / warmup_steps 
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * (1. + math.cos(math.pi * (current_step - warmup_steps) / (total_step - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_no_warmup(optimizer, sample_num, batch_idx, epoch, total_epoch, blr, start_step, min_lr=1e-6):
    """Decay the learning rate with half-cycle cosine"""
    total_step = total_epoch * sample_num
    current_step = (epoch - 1) * sample_num + batch_idx
    current_step = min(current_step + start_step, total_step)
    warmup_steps = int(total_step * 0.2)
    if current_step < warmup_steps:
        lr = blr
    else:
        lr = min_lr + (blr - min_lr) * 0.5 * (1. + math.cos(math.pi * (current_step - warmup_steps) / (total_step - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

def unnormalize(x):
    return (torch.clamp(x.float(), -1., 1.) + 1.0) / 2.0



def compute_loss(imgs_w, imgs, loss_percep):
    if len(imgs_w.shape) == 5: 
        imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
        imgs_w = imgs_w.view(-1, imgs_w.shape[2], imgs_w.shape[3], imgs_w.shape[4])
    return loss_percep((1 + imgs_w) / 2.0, (1 + imgs) / 2.0) / imgs_w.shape[0]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
