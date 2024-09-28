import text_preprocessor
import bigram_model
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
m = bigram_model.BigramLanguageModel(text_preprocessor.get_vocab_size())
model = m.to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(1000):
    # sample from the data
    xb, yb = text_preprocessor.get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    
    if steps%10 == 0: print(loss.item())

initial_in = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
generated_out = model.generate(idx= initial_in, max_new_tokens=100)[0].tolist()
print(text_preprocessor.decode(generated_out))

