import torch
import torch.nn as nn
from torch.nn import functional as F
from hyper_parameters import HyperParameters
hp = HyperParameters()

DROP_OUT = hp.DROP_OUT

class FeedForward(nn.Module):
    """ simple linear layer followed by non linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(DROP_OUT),
        )
    def forward(self, x):
        return self.net(x)