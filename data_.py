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
    def __init__(self, dir_,device,p_max=4.65,x_max=100,y_max=38.12,vx_max=2.04,vy_max=2.37,ro=1000):
        self.device   = device
        self.files_   = glob.glob(dir_)
        self.ro       = ro
        self.x_max    = x_max
        self.y_max    = y_max
        self.p_max    = p_max
        self.vx_max   = vx_max
        self.vy_max   = vy_max

    def __len__(self):
        return len(self.files_)


    def pressure_norm(self,x,u_inf):
        '''(Ian's density norm)
        Input: pressure matrix
          return: normilised pressure'''

        '''P/(ro_*|v|^2)'''

        x = x - np.mean(x) #remove offset
        new_x = x / (self.ro * u_inf ** 2)

        return new_x

    def __getitem__(self,idx):

        path_     = self.files_[idx]
        values_   = np.load(path_)['a']

        u_inf     = np.sqrt(values_[0][0,0]**2+values_[1][0,0]**2)

        angle = np.arctan(values_[1][0,0]/values_[0][0,0])
        angle *= 180/np.pi

        '''Normilize input velocity'''
        values_[0]  /= self.x_max
        values_[1]  /= self.y_max

        '''Normilize output velocity'''
        for i in [4,5]:
            values_[i] = values_[i]/u_inf

        '''Normilise pressure'''
        values_[3]     = self.pressure_norm(values_[3],u_inf)
        values_[3]    -= values_[3]*values_[2]

        '''Get cavitation number'''

        cav            = torch.full((1,128,128),cav_number(path_),device=self.device,dtype=torch.float32)
        flip_mask      = 1 - values_[2,:,:]
        cav            = cav*torch.tensor(flip_mask,device=self.device,dtype=torch.float32).view(1,128,128)

        '''Create model inputs'''
        Ux_Uy_mask     = torch.tensor(values_[:3,:,:],dtype=torch.float32,device=self.device)
        inputs_        = torch.cat([cav,Ux_Uy_mask],dim=0)

        '''create model outputs'''

        values_[3] /= self.p_max
        values_[4] /= self.vx_max
        values_[5] /= self.vy_max

        targets_       = torch.tensor(values_[3:,:,:],dtype=torch.float32,device=self.device)

        return inputs_, targets_, angle










