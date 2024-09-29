import torch
import torch.nn as nn
from torch.nn import functional as F
from attention import MultiHeadAttention
from feed_forward import FeedForward


class Block(nn.Module):
    """ transfoermer block: communication then computation """
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed//n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)               # 4 heads of attention each with size 8
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x