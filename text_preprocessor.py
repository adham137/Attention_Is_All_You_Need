import torch
torch.manual_seed(1337)

FILE_PATH = './Data/tiny_shakespeare.txt'   # data path
TRAIN_VAL_RATIO = 0.9                       # train validation split ratio
BLOCK_SIZE = 8                              # maximum contenxt length
BATCH_SIZE = 32                              # how many idependent sequence we are going to proccess in parallel
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}

def get_batch(split= 'train'):
    data = get_train() if split == 'train' else get_val()
    random_indicies = torch.randint(len(data)-BLOCK_SIZE-1, (BATCH_SIZE, ))         ## I added the -1, Keep Aware
    x = torch.stack([data[i:i+BLOCK_SIZE]for i in random_indicies])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1]for i in random_indicies])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y
    
def get_val():
    data = tokenize_dataset()
    n = int(TRAIN_VAL_RATIO*len(data))
    return data[n:]

def get_train():
    data = tokenize_dataset()
    n = int(TRAIN_VAL_RATIO*len(data))
    return data[:n]

def tokenize_dataset():
    return torch.tensor(encode(text), dtype=torch.long)

def encode(text):
    return [stoi[c] for c in text]

def decode(tokens):
    return ''.join([itos[t] for t in tokens])

def get_vocab_size():
    return len(chars)