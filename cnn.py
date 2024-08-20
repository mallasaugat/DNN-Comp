from imports import nn, torch


def corr2d(X, K):
    """2d cross-correlation"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y


class Conv2D(nn.Module):
    """Conv Layer"""

    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

def corr2d_multi_in(X,K):
    """Multi channel kernel"""
    return sum(d2l.corr2d(x, k), for x, k in zip(X,K))


def corr2d_multi_in_out(X,K):
    """Multiple Output Channels"""
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)

def corr2d_multi_int_out_1x1(X, K):
    c_i, h, w = X.shape 
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))


