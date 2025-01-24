import numpy as np

import torch
import torch.nn as nn

class ProjectionLoss(nn.Module):
    def __init__(self,num_type='real',reduction='mean',p=2, dim=1):
        super(ProjectionLoss,self).__init__()
        self.num_type = num_type
        self.reduction = reduction
        self.p=p
        self.dim=dim

        if self.num_type == 'complex':
            self.trans = torch.adjoint
        else:
            self.trans = torch.transpose
    
    def forward(self,Q,V):
        """
        Q: [batch_size,dim,k]
        V: [batch_size,dim,K] # K > k
        caculate sum(V-Q@Q*V)
        """

        assert Q.shape[-2] == V.shape[-2], f"Shape error! Q shape [{Q.shape[-2]}] must match V shape [{V.shape[-2]}] in dimension -2"

        Qt = self.trans(Q,-2,-1)
        QtV = torch.bmm(Qt,V)
        QQtV = torch.bmm(Q,QtV)

        result = V - QQtV
        norm = torch.norm(result,p=self.p,dim=self.dim)

        loss = torch.sum(norm,dim=-1)

        if self.reduction == 'mean':
            result = torch.mean(loss)
        elif self.reduction == 'sum':
            result = torch.sum(loss)
        
        return result
"""
def angle_between_subspaces(U, V):
    
    min_angle = np.pi / 2

    for u in U.T:
        u_norm = normalize(u)
        for v in v.T:
            v_norm = normalize(v)
            cos_angle = np.dot(u_norm,v_norm)
            angle = np.arccos(np.clip(cos_angle,-1.0,1.0))
            min_angle = min(min_angle,angle)
    
    return min_angle

def largest_principal_angle(U, V):
    singular_values = np.linalg.svd(np.dot(U, V.T), compute_uv=False)
    singular_values = np.clip(singular_values,0,1)
    angles = np.arccos(singular_values)
    return np.max(angles)
"""

class PrincipalAngle(nn.Module):
    """
    Caculate the angle between two subspace.
    type[Option]: biggest or smallest angle between two subspace
    
    WARINING: IF YOU NEED COMPUTE GRADIENT, PLEASE TURN OFF THE CLIP
    """
    def __init__(self, angle_type='biggest', reduction = 'mean', clip_value=True):
        super(PrincipalAngle,self).__init__()
        self.angle_type = angle_type
        self.reduction = reduction
        self.clip_value = clip_value
        self.compare = torch.max if self.angle_type == 'biggest' else torch.min

    def forward(self,Q,V):
        """
        Q,V: base vectors which make up the subspace
        """

        _, values, _ = torch.linalg.svd(torch.bmm(torch.transpose(Q,-2,-1),V))

        if self.clip_value:
            values = torch.clamp(values,min=-1,max=1)
        
        angles = torch.acos(values)
        
        angle,_ = self.compare(angles,dim=-1)
        
        if self.reduction == 'mean':
            result = torch.mean(angle)
        elif self.reduction == 'sum':
            result = torch.sum(angle)
        else:
            result = angle
        
        return result