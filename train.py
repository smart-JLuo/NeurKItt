import os
import json
import time
import tqdm
import numpy as np

import torch
import torch.nn as nn

from petsc4py import PETSc
from scipy.sparse import diags
from torch.nn import functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .model import FNO2d
from .loss import ProjectionLoss, PrincipalAngle
from .data_utils import move_to_cuda, collate_fn, mat_dataset


########################################################
# SETUP
########################################################

TRAIN_PATH = './data/helmholtz_train.pt'
VALID_PATH = './data/helmholtz_test.pt'

SAVE_PATH = './results/'

r = 250
width = 10
modes1 = 8
modes2 = 8
in_channel_size = 1
out_size = (r)**2
num_layers = 4
norm = True
COMPLEX = False
grid = True
num_type = 'complex' if COMPLEX else 'real'

ntrain = 1000
ntest = 200
k = 10
batch_size = 64
num_workers = 0

print("*"*20+" DATA LOADING "+"*"*20)

train_loader = DataLoader(mat_dataset(path=TRAIN_PATH,k=k), collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True, pin_memory=True)
valid_loader = DataLoader(mat_dataset(path=VALID_PATH,k=k), collate_fn=collate_fn, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

device = 'cuda:0'
lr = 1e-4
weight_decay = 1e-6
num_epoch = 10
iterations = num_epoch*len(train_loader)

model = FNO2d(modes1=modes1,
                modes2=modes2,
                width=width,
                in_channel_size=in_channel_size,
                out_size=out_size,
                resolution=r,
                num_layers=num_layers,
                num_space_size=k,
                norm=norm,
                grid=grid,
                COMPLEX=COMPLEX).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

## 
train_loss_fn = ProjectionLoss(num_type=num_type,reduction='mean')
test_loss_fn = PrincipalAngle()

train_loss_name = type(train_loss_fn).__name__
test_loss_name = type(test_loss_fn).__name__

########################################################
# TRAIN & SAVE
########################################################

print("*"*20+" TRAINING "+"*"*20)
for i in range(num_epoch):
    train_loss = []
    model.train()

    for j,batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        if device != 'cpu':
            batch_data = move_to_cuda(batch_data,device=device)

        input_data = {'inputs':batch_data['inputs']}

        label = batch_data['eig_vecs']
        output = model(**input_data)

        loss = train_loss_fn(output, label)
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_loss.append(loss.clone().detach().cpu())
    
    train_loss = torch.tensor(train_loss).mean()


    with torch.no_grad():
        eval_loss = []
        for k, batch_data in enumerate(valid_loader):

            batch_data = move_to_cuda(batch_data,device=device)

            input_data = {'inputs':batch_data['inputs']}

            label = batch_data['eig_vecs']

            output = model(**input_data)

            eval_loss.append(test_loss_fn(output,label).clone().detach().cpu())
        
        eval_loss = torch.tensor(loss).mean()
    
    print(f"Epoch:{i}   Train_loss({train_loss_name}):{train_loss.item():.3f}\tEval_loss({test_loss_name}):{eval_loss.item():.3f}")

print("*"*20+" TESTING "+"*"*20)

with torch.no_grad():
    params = []
    preds = []
    test_loss = []
    for i, batch_data in enumerate(valid_loader):
        params.append(batch_data['inputs'])
        if device != 'cpu':
            batch_data = move_to_cuda(batch_data,device=device)

        input_data = {'inputs':batch_data['inputs']}
        label = batch_data['eig_vecs']

        output = model(**input_data)

        
        loss = test_loss_fn(output,label)
        test_loss.append(loss.clone().detach().cpu().item())

        preds.append(output.clone().detach().cpu())

    params = torch.cat(params,dim=0)
    preds = torch.cat(preds,dim=0)

print(f"Test_loss({test_loss_name}):{torch.tensor(loss).mean():.3f}")

model_save_path = SAVE_PATH

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

torch.save(model.state_dict(), SAVE_PATH+'model.pt')
saving_dict = {'params':params.squeeze(),
            'preds':preds}
torch.save(saving_dict, SAVE_PATH+'preds.pt')