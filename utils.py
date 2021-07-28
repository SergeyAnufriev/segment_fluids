import torch

L1   = torch.nn.L1Loss(reduction='none')

def L1_(output_,target_):

    '''Input: output_, target_ with shape [1 * C * HIGHT * WIDTH],
     Output: L1 loss per channel'''
    diff_  = L1(output_,target_)

    return diff_.mean(dim=[0,2,3])


def relative_error(output_,target_,mask_,alpha=0.1,eps=1e-6):

    '''Input:  output_, target_ with shape [1 * C * HIGHT * WIDTH] and mask [1 * HIGHT *WIDTH]
       Output: relative error with shape 1 * C '''

    '''Number of channels vary depending on if void fraction is calculated or not'''
    n_channels    = target_.shape[1]
    diff_         = abs(target_-output_)

    '''mask pixels inside the airfoil'''
    masked_diff_  = torch.masked_select(diff_,~(mask_.bool()))
    masked_target = torch.masked_select(abs(target_),~(mask_.bool()))+eps

    error         = masked_diff_/masked_target
    l             = len(error)
    error         = error.view(n_channels,int(l/n_channels))
    error_indic_  = error>alpha

    n_pixels_outside = 128**2-mask_.sum()

    return error_indic_.sum(dim=1)/n_pixels_outside


def model_test(model,test_data_loader,n_var):

    '''Input: model to be evaluated on test set of data,
    test data set, n_vars - number of models output channels'''

    model.eval()

    L1_result = torch.zeros(size=(1,n_var))
    R_Error   = torch.zeros(size=(1,n_var))

    for input_,target_ in test_data_loader:

        output_   = model(input_)
        L1_result+= L1_(output_,target_)
        R_Error  += relative_error(output_,target_,input_[:,3,:,:])

    return L1_result/len(test_data_loader), R_Error/len(test_data_loader)

