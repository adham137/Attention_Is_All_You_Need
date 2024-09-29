import text_preprocessor
import bigram_model
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-2
MAX_ITERS = 1000                # maximum number of iterations 
EVAL_INTERVAL = 10             # evaluate the model after how many iterations ?
EVAL_ITERS = 10                # evaluate the model for how many iterations


@torch.no_grad()
def estimate_losses():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for iter in range(EVAL_ITERS):

            xb, yb = text_preprocessor.get_batch(split)
            # evaluate the loss
            logits, loss = model(xb, yb)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



m = bigram_model.BigramLanguageModel(text_preprocessor.get_vocab_size())
model = m.to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for iter in range(MAX_ITERS):

    # every once in a while evaluate the loss on the training and validation set
    if iter%EVAL_INTERVAL == 0:
        losses = estimate_losses()
        print(f"step {iter}: training loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample from the data
    xb, yb = text_preprocessor.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


initial_in = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
generated_out = model.generate(idx= initial_in, max_new_tokens=100)[0].tolist()
print(text_preprocessor.decode(generated_out))



