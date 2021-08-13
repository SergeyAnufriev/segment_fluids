import numpy as np
import random
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import torch
import torch.nn as nn
import wandb

'''Reference modules'''
from ref_unet import TurbNetG,weights_init

'''Custom modules'''
from data_ import Dataset_
from utils import relative_error

dir_train       = r'/content/gdrive/MyDrive/train_hydro_paper/train/*npz'
dir_test        = r'/content/gdrive/MyDrive/train_hydro_paper/train/*npz'

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

'''First params estimates'''
n_var      = 3
epochs     = 40
batch_size = 10
lrG        = 0.0006
iterations = 10000
minLR = lrG*0.1
maxLR = lrG

train_data_loader = DataLoader(Dataset_(dir_train,device,ro=1.),\
                               batch_size=batch_size,drop_last=True,worker_init_fn=seed_worker,generator=g)
test_data_loader  = DataLoader(Dataset_(dir_test,device,ro=1.),\
                               batch_size=1,worker_init_fn=seed_worker,generator=g)

def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name,
                         nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))  # Note: old default was nearest neighbor
        # reduce kernel size by one for the upsampling (ie decoder part)
        block.add_module('%s_tconv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=(size - 1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block


# generator model
class TurbNetG(nn.Module):
    def __init__(self, channelExponent=5, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer2b = blockUNet(channels * 2, channels * 2, 'layer2b', transposed=False, bn=True, relu=False,
                                 dropout=dropout)
        self.layer3 = blockUNet(channels * 2, channels * 4, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout)
        # note the following layer also had a kernel size of 2 in the original version (cf https://arxiv.org/abs/1810.08217)
        # it is now changed to size 4 for encoder/decoder symmetry; to reproduce the old/original results, please change it to 2
        self.layer4 = blockUNet(channels * 4, channels * 8, 'layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=4)  # note, size 4!
        self.layer5 = blockUNet(channels * 8, channels * 8, 'layer5', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=2, pad=0)
        self.layer6 = blockUNet(channels * 8, channels * 8, 'layer6', transposed=False, bn=False, relu=False,
                                dropout=dropout, size=2, pad=0)

        # note, kernel size is internally reduced by one now
        self.dlayer6 = blockUNet(channels * 8, channels * 8, 'dlayer6', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer5 = blockUNet(channels * 16, channels * 8, 'dlayer5', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer4 = blockUNet(channels * 16, channels * 4, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer3 = blockUNet(channels * 8, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer2b = blockUNet(channels * 4, channels * 2, 'dlayer2b', transposed=True, bn=True, relu=True,
                                  dropout=dropout)
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels * 2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2b = self.layer2b(out2)
        out3 = self.layer3(out2b)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2b = torch.cat([dout3, out2b], 1)
        dout2b = self.dlayer2b(dout3_out2b)
        dout2b_out2 = torch.cat([dout2b, out2], 1)
        dout2 = self.dlayer2(dout2b_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1

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