import torch
import torch.nn as nn
from torch.nn import functional as F

N_EMBED = 32        # number of embedding dimensions
# HEAD_SIZE = 16      # size of attention head
BLOCK_SIZE = 8      # maximum contenxt length
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class FeedForward(nn.Module):
    """ simple linear layer followed by non linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
        )
    def forward(self, x):
        return self.net(x)