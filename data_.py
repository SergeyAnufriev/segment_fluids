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


class Dataset_(Dataset):
    def __init__(self, dir_,device):
        self.device   = device
        self.files_   = glob.glob(dir_)

    def __len__(self):
        return len(self.files_)

    @staticmethod
    def pressure_norm(x):
        '''(Ian's density norm)
        Input: pressure matrix
          return: normilised pressure'''

        '''P/(ro_*|v|^2)'''
        x    = x/(1000*2**2)
        mean = np.mean(x)

        return x - mean

    def __getitem__(self,idx):

        path_     = self.files_[idx]
        values_   = np.load(path_)['a']

        '''Normilize velocity'''
        for i in [0,1,4,5]:
            values_[i] = values_[i]/2

        '''Normilise pressure'''
        values_[3]     = self.pressure_norm(values_[3])

        '''Get cavitation number'''
        cav            = torch.full((1,128,128),cav_number(path_),device=self.device)
        cav            = cav*torch.tensor(values_[2,:,:],device=self.device).view(1,128,128)

        '''Create model inputs'''
        Ux_Uy_mask     = torch.tensor(values_[:3,:,:],dtype=torch.float32,device=self.device)
        inputs_        = torch.cat([cav,Ux_Uy_mask],dim=0)

        '''create model outputs'''
        targets_       = torch.tensor(values_[3:,:,:],dtype=torch.float32,device=self.device)

        return inputs_, targets_








