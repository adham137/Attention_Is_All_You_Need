import torch
import torch.nn as nn
from torch.nn import functional as F
from hyper_parameters import HyperParameters
hp = HyperParameters()


N_EMBED = hp.N_EMBED
BLOCK_SIZE = hp.BLOCK_SIZE
DROP_OUT = hp.DROP_OUT

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, HEAD_SIZE):
        super().__init__()
        self.key = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.do = nn.Dropout(DROP_OUT)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.do(wei)
        # perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention """
    def __init__(self,  num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        n_embed = num_heads*head_size
        self.proj = nn.Linear(n_embed, n_embed)
        self.do = nn.Dropout(DROP_OUT)
    def forward(self,x):
        out = torch.cat([h(x)for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.do(out)
        return out
