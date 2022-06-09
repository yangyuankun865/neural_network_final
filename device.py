import os
os.environ['CUDA_VISIBLE_DEVICES']='5'
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("current device",torch.cuda.current_device())