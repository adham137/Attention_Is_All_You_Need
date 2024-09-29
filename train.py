import text_preprocessor
import bigram_model
import torch
from hyper_parameters import HyperParameters
hp = HyperParameters()

DEVICE = hp.DEVICE
LEARNING_RATE = hp.LEARNING_RATE
MAX_ITERS = hp.MAX_ITERS
EVAL_INTERVAL = hp.EVAL_INTERVAL
EVAL_ITERS = hp.EVAL_ITERS
OUTPUT_FILE_PATH = hp.OUTPUT_FILE_PATH
MAX_NEW_TOKENS = hp.MAX_NEW_TOKENS

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
    if iter%EVAL_INTERVAL == 0 :
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
generated_out = model.generate(idx= initial_in, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
decoded_string = text_preprocessor.decode(generated_out)
print(decoded_string)
with open(OUTPUT_FILE_PATH, 'w') as file:
    file.write(decoded_string)



