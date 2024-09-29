import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_block import Block

N_EMBED = 32        # number of embedding dimensions
BLOCK_SIZE = 8                              # maximum contenxt length
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HEAD_SIZE = 4       # attention head size

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(
            Block(N_EMBED, n_heads=HEAD_SIZE), 
            Block(N_EMBED, n_heads=HEAD_SIZE), 
            Block(N_EMBED, n_heads=HEAD_SIZE), 
        )
        self.lm_head = nn.Linear(N_EMBED, vocab_size)                   # converts the embeddings into vocab size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embeddings = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T, C)
        x = token_embeddings + pos_embeddings   # addition of both embeddings (B,T,C)
        x = self.blocks(x)
        logits = self.lm_head(x)            # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # reshaping the logits shape to calculate the loss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            # crop the index for safety
            idx_cond = idx[:, -BLOCK_SIZE:] 
            # get predictions
            logits, loss  = self(idx_cond)
            # focus on last timestep
            logits = logits[:, -1, :]               # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)       # (B, C)
            # sample from the probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append the sampled index to the sequence
            idx = torch.cat((idx, idx_next), dim=1)     # (B, T+1)
        return idx