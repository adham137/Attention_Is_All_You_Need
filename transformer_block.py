import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import MultiHeadAttention
from feed_forward import FeedForward

N_EMBED = 32        # number of embedding dimensions
# HEAD_SIZE = 16      # size of attention head
BLOCK_SIZE = 8      # maximum contenxt length
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Block(nn.Module):
    """ transfoermer block: communication then computation """
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed//n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)               # 4 heads of attention each with size 8
        self.ffwd= FeedForward(N_EMBED)
    def forward(self, x):
        x = x + self.sa_heads(x)
        x = x + self.ffwd(x)
        return x