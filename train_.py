import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import wandb

'''Reference modules'''
from ref_unet import TurbNetG,weights_init

'''Custom modules'''
from data_ import Dataset_
from utils import relative_error

'''First params estimates'''
n_var      = 3
epochs     = 16
batch_size = 10
lrG        = 0.0006

'''Fix random seed to make results reproducible'''
seed = 356
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

'''Fix dataloader seed https://pytorch.org/docs/stable/notes/randomness.html'''

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dir_train       = r'/content/gdrive/MyDrive/train_hydro_paper/train/*npz'
dir_test        = r'/content/gdrive/MyDrive/train_hydro_paper/train/*npz'


train_data_loader = DataLoader(Dataset_(dir_train,device,ro=1.,u_inf=99.99320244830442),\
                               batch_size=batch_size,drop_last=True,worker_init_fn=seed_worker,generator=g)
test_data_loader  = DataLoader(Dataset_(dir_test,device,ro=1.,u_inf=99.99320244830442),\
                               batch_size=1,worker_init_fn=seed_worker,generator=g)

'''Initialise model'''
'''TO  DO PARAM Grid'''
net    = TurbNetG()

''' Apply weight intialisation strategy'''
net.apply(weights_init)
net.to(device)

L1  = torch.nn.L1Loss()
'''Optimizer settings'''
opt = optim.Adam(net.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

#scheduler = optim.lr_scheduler.LambdaLR(opt,lr_lambda=computeLR)


'''Save actual vs predicted images'''

columns = ['epoch', 'id', 'Pressure', 'V_x', 'V_y']
test_table = wandb.Table(columns=columns)

start_time = time.time()

for epoch in range(epochs):

    '''Training loop'''
    net.train()
    train_L1_accum = 0

    for input_, target_ in train_data_loader:
        opt.zero_grad()
        output_ = net(input_[:, 1:, :, :])
        loss = L1(output_, target_)
        loss.backward()
        train_L1_accum += loss.item()

        opt.step()

    '''Apply learning rate decay'''
    # scheduler.step()

    wandb.log({'Train_loss': train_L1_accum / len(train_data_loader), 'Epoch': epoch})

    '''Validation Loop'''
    net.eval()
    test_loss_accum = 0

    '''Relative error per channel'''
    R_Error = torch.zeros(size=(1, n_var), device=device)

    counter = 0
    for input_, target_ in test_data_loader:

        counter += 1
        output_ = net(input_[:, 1:, :, :])
        loss = L1(output_, target_)
        test_loss_accum += loss.item()
        R_Error += relative_error(output_, target_, input_[:, 3, :, :])

        '''Log every 30th Predicted vs Actual Channel Image'''
        if counter % 30 == 0:

            images = []

            for i in range(n_var):
                x = np.reshape(target_[:, i, :, :].cpu().numpy(), (128, 128)).T
                y = np.reshape(output_[:, i, :, :].detach().cpu().numpy(), (128, 128)).T
                img_data = wandb.Image(np.concatenate((x, y), axis=1))
                images.append(img_data)

            test_table.add_data(epoch, counter, *images)

    wandb.log({'Channels': test_table})

    '''Log tets loop results'''
    E_P, E_V_x, E_V_y = R_Error[0] / len(test_data_loader)

    wandb.log({'Test_loss': test_loss_accum / len(test_data_loader), 'Epoch': epoch, \
               'E_P': E_P, 'E_V_x': E_V_x, 'E_V_y': E_V_y})

end_time = time.time()
total_time = end_time - start_time
print("Time: ", total_time)