from imports import F, nn, torch


def init_normal(module):
    """Initialize weight parameters as Gaussian Random variables with sd 0.01"""
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)


def init_constant(module):
    """Initialize weights with constant"""
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)


def init_xavier(module):
    """Xavier Initialization"""
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)


def apply_init(self, inputs, init=None):
    """Lazy Init"""
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
