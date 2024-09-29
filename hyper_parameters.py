import torch
class HyperParameters:
    def __init__(self):
        self.FILE_PATH = './Data/tiny_shakespeare.txt'   # data path
        self.TRAIN_VAL_RATIO = 0.9                       # train validation split ratio
        self.BLOCK_SIZE = 256                              # maximum sequence length
        self.BATCH_SIZE = 64                             # how many idependent sequence we are going to proccess in parallel

        self.OUTPUT_FILE_PATH = './output.txt'
        self.MAX_NEW_TOKENS = 10000

        self. DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.N_EMBED = 192                                # number of embedding dimensions
        self.N_BLOCK = 6                                 # number of transformer blocks
        self.N_HEAD = 6                                  # number of attention heads at each transfomer block
        self.DROP_OUT = 0.2

        self.LEARNING_RATE = 3e-4
        self.MAX_ITERS = 1250                            # maximum number of iterations 
        self.EVAL_INTERVAL = 50                           # evaluate the model after how many iterations ?
        self.EVAL_ITERS = 20                             # evaluate the model for how many iterations

