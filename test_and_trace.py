from ref_unet import TurbNetG,weights_init
from data_ import Dataset_
from torch.utils.data import DataLoader
import torch
from utils import model_test
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import wandb


os.environ['KMP_DUPLICATE_LIB_OK']='True'

dir_   = r'C:\Users\zcemg08\PycharmProjects\segment_fluids\data\train_hydro_2021\*.npz'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()


net    = TurbNetG()
net.apply(weights_init)
data_loader = DataLoader(Dataset_(dir_,device),batch_size=1)

from utils import relative_error

for input_,target_ in data_loader:
    break

print(input_[:,0,:,:])


