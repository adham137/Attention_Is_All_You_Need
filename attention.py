import torch
import torch.nn as nn
from torch.nn import functional as F

N_EMBED = 32        # number of embedding dimensions
# HEAD_SIZE = 16      # size of attention head
BLOCK_SIZE = 8      # maximum contenxt length
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, HEAD_SIZE):
        super().__init__()
        self.key = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.query = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.value = nn.Linear(N_EMBED, HEAD_SIZE, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
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
    def forward(self,x):
        out = torch.cat([h(x)for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
