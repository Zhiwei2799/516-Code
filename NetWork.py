import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class PortfolioModel(nn.Module):
    def __init__(self, r, P, cov, Lambda, Delta, K, lb, ub,Rf,batch_size):
        super(PortfolioModel, self).__init__()
        self.K = K
        self.Rf = Rf
        self.batch_size = batch_size
        self.stack_layers = nn.ModuleList()
        for i in range(K):
            self.stack_layers.append(StackLayer(r, P, cov, Lambda, Delta, K, lb, ub,Rf,batch_size))
    
    def forward(self,inputs):
#         print('initial portfolio wealth: 1')
        for i in range(self.K):
#             print('subnetwork {}'.format(i))
            inputs = self.stack_layers[i](inputs)
            if i == 1:
                score = (inputs.detach().numpy().mean()*np.exp(0.03/40)**-1 - 2)**2
                return inputs.detach().numpy().mean()
        return inputs

class Subnetwork(nn.Module):
    def __init__(self, r, P, cov, Lambda, Delta, K, lb, ub,Rf, batch_size):
        super(Subnetwork, self).__init__()
        self.Rf = Rf
        self.lb = lb
        self.ub = ub
        epsilon = np.random.multivariate_normal(mean=np.zeros(int(P)), cov=np.eye(int(P))) 
        self.R = np.zeros((batch_size, P))
        for b in range(batch_size):          
                epsilon = np.random.multivariate_normal(mean=np.zeros(P), cov=np.eye(P))
                self.R[b, :] = np.exp((r * np.ones(P) + cov @ Lambda - (1/2) * np.diag(cov @ cov.T)) * Delta +
                    np.sqrt(Delta) * cov @ epsilon) - Rf * np.ones(P)
        self.R = torch.Tensor(self.R)
        self.subnetwork = nn.Sequential(
            nn.Linear(5, 5, bias=False),
            nn.Tanh(),
            nn.Linear(5, 5, bias=False),
            nn.Tanh(),
            nn.Linear(5, 5, bias=False),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        state_variable = x 
        out = self.R 
        weights = self.subnetwork(out)
#         print("weights before rebalancing:", weights.detach().numpy())
        
        output = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in weights])
#         print('weights after rebalancing',output.detach().numpy())
#         print('self.R', self.R.shape)
#         print('self.Rf',self.Rf.shape)

        updated_state_variable = torch.stack([torch.sum(state_variable[i] * output[i] * (1+self.R[i])* self.Rf) for i in range(output.shape[0])])
#         print('returns',self.R+1)
#         print('Rf',self.Rf)
#         print(((output * (self.R + Rf)).sum())
#         print('new portfolio wealth:', updated_state_variable.detach().numpy())
        return updated_state_variable 

    def rebalance(self, weight, lb, ub):
        old = weight
        new = torch.clamp(old, lb, ub)
        while True:
            leftover = (old - new).sum()
            other_stocks = new[torch.where(new != ub)[0]]
            reassign = leftover * (other_stocks/other_stocks.sum())
            new[torch.where(new != ub)[0]] += reassign 
            old = new
            if len(torch.where(new > ub)[0]) == 0:
                break
            else:
                new = torch.clamp(old, lb, ub)
        return new
    
class StackLayer(nn.Module):
    def __init__(self, r, P, cov, Lambda, Delta, K, lb, ub,Rf,batch_size):
        super(StackLayer, self).__init__()
        self.blocks = nn.ModuleList([Subnetwork(r, P, cov, Lambda, Delta, K, lb, ub,Rf,batch_size)])   
    def forward(self, inputs):
        outputs = inputs 
        for block in self.blocks:
            outputs = block(outputs)
        return outputs
