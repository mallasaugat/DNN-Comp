from numpy import add

from imports import nn, torch
from utility import Trainer, add_to_class


def cpu():
    """Get the CPU device."""
    return torch.device("cpu")


def gpu(i=0):
    """Get a GPU device."""
    return torch.device(f"cuda:{i}")


def num_gpus():
    """Get the number of available GPUs"""
    return torch.cuda.device_count()


def try_gpus(i=0):
    """Returns gpu with number if exist, else returns cpu()"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()


def try_all_gpus():
    """Return all available GPUs, or [CPU(), ] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]


@add_to_class(Trainer)
def __init__(self, max_epochs, num_gpu=0, gradient_clip_val=0):
    """Set gpus in init of trainer class"""
    self.save_hyperparameters()
    self.gpus = [gpu(i) for i in range(min(num_gpu, num_gpus()))]


@add_to_class(Trainer)
def prepare_batch(self, batch):
    """Set gpus in batch module of Trainer class"""
    if self.gpus:
        batch = [a.to(self.gpus[0]) for a in batch]
    return batch


@add_to_class(Trainer)
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        model.to(self.gpus[0])
    self.model = model
