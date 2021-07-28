from torch.utils.data import Dataset
import torch
import numpy as np
import glob

dir_ = r'C:\Users\zcemg08\PycharmProjects\segment_fluids\data\train_hydro_2021'

def cav_number(path):

    '''Input:  file location
       Output: cavitation number'''

    file  = path.split('\\')[-1]
    props = file.split('_')
    cav   = float(props[-2]) / 100

    return cav


def angle_(path):
    '''Input : file name
      Output : angle of attack in degress'''
    file = path.split('\\')[-1]
    props = file.split('_')

    return  np.arctan(float(props[-4])/ float(props[-5]))*180/np.pi



class Dataset_(Dataset):
    def __init__(self, dir_,device,ro=1000,u_inf=2.):
        self.device   = device
        self.files_   = glob.glob(dir_)
        self.u_inf    = u_inf
        self.ro       = ro

    def __len__(self):
        return len(self.files_)


    def pressure_norm(self,x):
        '''(Ian's density norm)
        Input: pressure matrix
          return: normilised pressure'''

        '''P/(ro_*|v|^2)'''
        x    = x/(self.ro*self.u_inf**2)
        mean = np.mean(x)

        return x - mean

    def __getitem__(self,idx):

        path_     = self.files_[idx]
        values_   = np.load(path_)['a']

        '''Normilize velocity'''
        for i in [0,1,4,5]:
            values_[i] = values_[i]/self.u_inf

        '''Normilise pressure'''
        values_[3]     = self.pressure_norm(values_[3])

        '''Get cavitation number'''
        cav_n          = cav_number(path_)
        cav            = torch.full((1,128,128),cav_n,device=self.device,dtype=torch.float32)
        flip_mask      = 1 - values_[2,:,:]
        cav            = cav*torch.tensor(flip_mask,device=self.device,dtype=torch.float32).view(1,128,128)

        '''Angle of attack'''
        alpa           = angle_(path_)

        '''Create model inputs'''
        Ux_Uy_mask     = torch.tensor(values_[:3,:,:],dtype=torch.float32,device=self.device)
        inputs_        = torch.cat([cav,Ux_Uy_mask],dim=0)

        '''create model outputs'''
        targets_       = torch.tensor(values_[3:,:,:],dtype=torch.float32,device=self.device)

        return inputs_, targets_, cav_n,alpa








