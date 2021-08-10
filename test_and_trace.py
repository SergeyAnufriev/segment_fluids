from data_ import Dataset_
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os


os.environ['KMP_DUPLICATE_LIB_OK']='True'

dir_   = r'C:\Users\zcemg08\PycharmProjects\segment_fluids\data\paper\train_\*.npz'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_loader = DataLoader(Dataset_(dir_,device,ro=1.,u_inf=100),batch_size=1)

p_abs  = []
vx_abs = []
vy_abs = []

for _, target_ in data_loader:
    p_abs.append(torch.max(abs(target_[0,0,:,:])).data.numpy())
    vx_abs.append(torch.max(abs(target_[0,1,:,:])).data.numpy())
    vy_abs.append(torch.max(abs(target_[0,2,:,:])).data.numpy())


fig, (ax1, ax2,ax3) = plt.subplots(1, 3,figsize=(10, 4))

ax1.hist(p_abs)
ax1.set_xlabel('Maximum absolute pressure')
ax1.set_ylabel('Count')
ax1.set_title('Pressure')

ax2.hist(vx_abs)
ax2.set_xlabel('Maximum absolute V_X')
ax2.set_ylabel('Count')
ax2.set_title('V_X')

ax3.hist(vy_abs)
ax3.set_xlabel('Maximum absolute V_Y')
ax3.set_ylabel('Count')
ax3.set_title('Maximum absolute V_Y')

plt.show()







