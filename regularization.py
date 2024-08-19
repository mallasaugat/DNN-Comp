from imports import nn, torch
from utility import Classifier, HyperParameters


class WeightDecay(HyperParameters):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD(
            [
                {"params": self.net.weight, "weight_decay": self.wd},
                {"params": self.net.bias},
            ],
            lr=self.lr,
        )


class DropoutMLP(Classifier):
    """Dropout for MLP-Classifier"""

    def __init__(
        self, num_outputs, num_hiddens_1, num_hiddens_2, droupout_1, droupout_2, lr
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens_1),
            nn.ReLU(),
            nn.Dropout(droupout_1),
            nn.LazyLinear(num_hiddens_2),
            nn.ReLU(),
            nn.Dropout(droupout_2),
            nn.LazyLinear(num_outputs),
        )
