import tqdm
import json
import math
import torch
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

import torch
import math

class GaussianRF(object):

    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumers.transpose(0,1)
            k_y = wavenumers

            self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumers.transpose(1,2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0,2)

            self.sqrt_eig = (size**3)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):

        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)
        coeff = self.sqrt_eig * coeff

        return torch.fft.ifftn(coeff, dim=list(range(-1, -self.dim - 1, -1))).real


def build(coef):
    K = coef.shape[0]
    coef = coef.numpy()
    s = K - 2

    diag_list = []
    off_diag_list = []
    for j in range(1, K-1):
        diag_values = np.array([
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0])),
            0.5 * (coef[0:K-2, j] + coef[1:K-1, j]) + 0.5 * (coef[2:K, j] + coef[1:K-1, j]) + \
            0.5 * (coef[1:K-1, j-1] + coef[1:K-1, j]) + 0.5 * (coef[1:K-1, j+1] + coef[1:K-1, j]),
            np.concatenate((-0.5 * (coef[1:K-2, j] + coef[2:K-1, j]),[0]))
        ])
        diag_list.append(diag_values)

        if j != K-2:
            off_diag = -0.5 * (coef[1:K-1, j] + coef[1:K-1, j+1])
            off_diag_list.append(off_diag)

    diag_output = np.concatenate(diag_list,axis=1)
    off_diag_output = np.concatenate(off_diag_list,axis=0)
    A = (diags(diag_output,[-1,0,1],(s**2,s**2)) + diags((off_diag_output,off_diag_output),[-(K-2),(K-2)],(s**2,s**2))) * (K-1)**2

    return A

if __name__ == "__main__":
    s = 402
    k = 25
    edge = s-2
    tol= 1e-14

    train_resolution=s

    train_num, test_num = 6000,1200
    num_examples = train_num + test_num

    GRF = GaussianRF(2, s, alpha=2, tau=3, device='cpu')
    f = np.ones([s,s])

    x=[]
    A=[]
    eig_vecs=[]
    eig_vals=[]
    w = np.exp(GRF.sample(num_examples+1))
    
    pbar = tqdm.tqdm(desc=f"Darcy: {s}",total=num_examples,position=0,leave=True)

    for i in range(num_examples):
        pbar.update()
        wi = w[i]

        x.append(wi.unsqueeze(0))

        A1 = build(wi)

        eig_val, eig_vec = eigs(A1, k, sigma=0, which="LM", tol=tol)

        eig_vecs.append(torch.tensor(eig_vec.real, dtype=torch.float32).unsqueeze(0))
        eig_vals.append(torch.tensor(eig_val.real, dtype=torch.float32).unsqueeze(0))

    pbar.close()

    x_train = x[:train_num]
    x_test = x[train_num:]

    y_train = eig_vecs[:train_num]
    y_test = eig_vecs[train_num:]

    val_train = eig_vals[:train_num]
    val_test = eig_vals[train_num:]
    
    train_data = {"params":torch.cat(x_train,dim=0),
                  "eig_vecs":torch.cat(y_train,dim=0),
                  "eig_vals":torch.cat(val_train,dim=0),}
    
    test_data = {"params":torch.cat(x_test,dim=0),
                 "eig_vecs":torch.cat(y_test,dim=0),
                 "eig_vals":torch.cat(val_test,dim=0),}

    torch.save(train_data,f"darcy_train.pt")

    torch.save(test_data,f"darcy_test.pt")