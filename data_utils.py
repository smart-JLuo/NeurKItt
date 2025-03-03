import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def move_to_cuda(sample, device=None):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor,device):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value,device) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x,device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x,device) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample,device)

# def 

def collate_fn(batch_data):
    inputs = []
    eig_vecs = []

    for _,data in enumerate(batch_data):
        inputs.append(data['input'])
        eig_vecs.append(data['eig_vec'])

    return {'inputs':torch.cat(inputs,dim=0),
            'eig_vecs':torch.cat(eig_vecs,dim=0)
    }

class mat_dataset(Dataset):
    def __init__(self,
                 path,
                 k,
                 channel_dim=3,
                 test_num=None
                 ):
        super(mat_dataset, self).__init__()
        print(f"loading data from {path}")
        data = torch.load(path)


        if channel_dim > 0:
            self.input = data['params'].unsqueeze(channel_dim)
        else:
            self.input = data['params']
        

        if k>0:
            subspace_shape = data['eig_vecs'].shape
            if subspace_shape[-1]<k:
                print(f"Training subspace not match, got model {k} but dataset {subspace_shape[-1]}")
            self.eig_vecs = data['eig_vecs'][...,:,:k]
        else:
            print(f"K not assigned appropriate, using full size")
            self.eig_vecs = data['eig_vecs']


        print(f"Training subspace size is {self.eig_vecs.shape[-1]}")

        self.mat_dim = self.input.shape[-1]

        self.num_examples = self.input.shape[0] if test_num is None else test_num

    def __getitem__(self,index):
        return {'input':self.input[index].unsqueeze(0),
                'eig_vec':self.eig_vecs[index].unsqueeze(0),}

    def __len__(self):
        return self.num_examples