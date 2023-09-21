import torch

device = torch.device('cuda')
dataset = "gowalla_small"
model_name = "lgn"
test_u_batch_size = 100
topks = [20]
bpr_batch_size = 2048
tensorboard = 1
comment = "lightgcn"

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
