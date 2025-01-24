import torch
import numpy as np
import torch.nn as nn

from petsc4py import PETSc
from scipy.sparse import diags
from torch.nn import functional as F

########################################################
# 1. MODEL
########################################################

def CReLU(x):
    return torch.complex(F.relu(x.real), F.relu(x.imag))

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ComplexInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ComplexInstanceNorm2d, self).__init__()
        self.norm_real = nn.InstanceNorm2d(num_features, eps=eps)
        self.norm_imag = nn.InstanceNorm2d(num_features, eps=eps)

    def forward(self, x):
        real = self.norm_real(x.real)
        imag = self.norm_imag(x.imag)
        return torch.complex(real, imag)


class ComplexConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ComplexConv2d, self).__init__()

        scale = 1 / (in_channels * out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dtype=torch.cfloat)
        self.stride = stride
        self.padding = padding


    def forward(self, x):
        output = self.conv(x)
        return output


class ComplexConv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ComplexConv1d, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dtype=torch.cfloat)


        self.stride = stride
        self.padding = padding

    def forward(self, x):
        output = self.conv(x)
        return output


class SpectralConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes1,
                 modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv2d_Complex(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 modes1,
                 modes2):
        super(SpectralConv2d_Complex, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1), dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft[:, :, -self.modes1:, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, -self.modes2:], self.weights3)
        out_ft[:, :, :self.modes1, -self.modes2:] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, -self.modes2:], self.weights3)

        #Return to physical space
        x = torch.fft.ifft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 COMPLEX = False):
        super(MLP, self).__init__()
        self.COMPLEX = COMPLEX

        if COMPLEX:
            self.mlp1 = ComplexConv2d(in_channels, mid_channels, 1)
            self.mlp2 = ComplexConv2d(mid_channels,out_channels, 1)
            self.act_fun = CReLU
        else:
            self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
            self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
            self.act_fun = F.gelu


    def forward(self, x):
        x = self.mlp1(x)
        x = self.act_fun(x)
        x = self.mlp2(x)
        return x


class MLP1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 COMPLEX = False):
        super(MLP, self).__init__()
        self.COMPLEX = COMPLEX

        if COMPLEX:
            self.mlp1 = ComplexConv1d(in_channels, mid_channels, 1)
            self.mlp2 = ComplexConv1d(mid_channels,out_channels, 1)
            self.act_fun = CReLU
        else:
            self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
            self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)
            self.act_fun = F.gelu


    def forward(self, x):
        x = self.mlp1(x)
        x = self.act_fun(x)
        x = self.mlp2(x)
        return x


class FourierLayer(nn.Module):
    def __init__(self,
                 width,
                 modes1,
                 modes2,
                 act = True,
                 norm = False,
                 COMPLEX = False):
        super(FourierLayer,self).__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.act = act

        if COMPLEX:
            self.conv = SpectralConv2d_Complex(self.width, self.width, self.modes1, self.modes2)
        else:
            self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)

        self.mlp = MLP(self.width, self.width, self.width, COMPLEX=COMPLEX)
        self.norm = norm

        if COMPLEX:
            self.w = ComplexConv2d(self.width, self.width, 1)
            self.instance_norm = ComplexInstanceNorm2d(self.width)
            self.act_fun = CReLU
        else:
            self.w = nn.Conv2d(self.width, self.width, 1)
            self.instance_norm = nn.InstanceNorm2d(self.width)
            self.act_fun = F.gelu
        
        if not self.act:
            self.act_fun = Identity()
    
    def forward(self,x):

        if self.norm:
            x1 = self.instance_norm(self.conv(self.instance_norm(x)))
        else:
            x1 = self.conv(self.instance_norm(x))

        x1 = self.mlp(x1)
        x2 = self.w(x)
        x = x1 + x2
        x = self.act_fun(x)
        return x

class FNO2d(nn.Module):
    def __init__(self,
                 modes1,
                 modes2,
                 width,
                 in_channel_size,
                 out_size,
                 resolution,
                 num_layers = 4,
                 num_space_size = 10,
                 norm = True,
                 grid = False,
                 COMPLEX = False,
                 periodic = False):
        super(FNO2d, self).__init__()
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.grid = grid
        self.resolution = resolution
        self.in_channel_size = in_channel_size
        self.out_size = out_size
        self.periodic = periodic
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.norm = norm
        self.COMPLEX = COMPLEX
        self.num_layers = num_layers
        self.num_space_size = num_space_size

        self.padding = 8 if self.periodic else 0

        self.p = MLP(self.in_channel_size + 2, self.width, self.width, COMPLEX) if self.grid else MLP(self.in_size, self.width, self.width, COMPLEX)

        self.FourierLayers = nn.ModuleList([FourierLayer(self.width, self.modes1, self.modes2, act = True, norm = self.norm, COMPLEX = self.COMPLEX) \
                                            if (i+1)!=self.num_layers else FourierLayer(self.width, self.modes1, self.modes2, act = False, norm = self.norm, COMPLEX = self.COMPLEX) for i in range(self.num_layers)])

        self.q1 = MLP(self.width, self.num_space_size, self.width * 4, COMPLEX)

        # self.q2 = ComplexConv1d(self.resolution**2,self.out_size,1) if self.COMPLEX else nn.Conv1d(self.resolution**2,self.out_size,1)
        # self.q2 = ComplexConv1d(self.resolution**2,self.out_size,1) if self.COMPLEX else Identity()

        # If output size not meet the requires, consider cut the output rather than using linear mapping
        self.q2 = Identity() # DO NOT APPLY PROJECTION TO Q1 OUTPUT

    def forward(self, inputs, grid=None):

        if self.grid and grid == None:
            grid = self.get_grid(inputs.shape, inputs.device)

        if self.grid:
            inputs = torch.cat((inputs, grid), dim=-1)

        x = inputs.permute(0, 3, 1, 2)

        x = self.p(x)

        if self.padding > 0:
            x = F.pad(x, [0,self.padding, 0,self.padding])

        for layer in self.FourierLayers:
            x = layer(x)

        x = self.q1(x)

        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]

        x = x.permute(0, 2, 3, 1) # (batch_size, subspace_size, s, s) -> (batch_size, s, s, subspace_size)

        batch_size = x.shape[0]
        
        x = x.reshape(batch_size, -1, self.num_space_size) # (batch_size, s, s, subspace_size) -> (batch_size, s*s, subspace_size)
        assert x.shape[1] == self.resolution**2

        x = self.q2(x) # (batch_size, s*s, subspace_size) -> (batch_size, out_size, subspace_size)
        # x = x.permute(0,2,1)
        Q,R = torch.linalg.qr(x)

        return Q

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.cfloat) \
            if self.COMPLEX else torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float) 
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.cfloat) \
            if self.COMPLEX else torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float) 
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)