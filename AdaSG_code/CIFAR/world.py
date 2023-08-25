import torch

device = torch.device('cuda')
dataset = "gowalla"
model_name = "lgn"

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")
