import torch
import torch.nn as nn
from torch import Tensor


class PortfolioModel(nn.Module):
    def __init__(self, P, cov, K, lb, ub,Rf,batch_size,alpha,A):
        super(PortfolioModel, self).__init__()
        self.K = K
        self.Rf = Rf
        self.cov = cov
        self.batch_size = batch_size
        self.alpha = alpha 
        self.A = A
        self.stack_layers = nn.ModuleList()
        for i in range(K):
            self.stack_layers.append(StackLayer(P, cov, K, lb, ub,Rf,batch_size,alpha,A))
    
    def forward(self,input1, input2):
#         print('initial portfolio wealth: 1')
        for i in range(self.K):
#             print('subnetwork {}'.format(i))
            inputs = self.stack_layers[i](input1, input2)            
        return inputs

class Subnetwork(nn.Module):
    def __init__(self, P, cov, K, lb, ub,Rf, batch_size, alpha,A):
        super(Subnetwork, self).__init__()
        self.Rf = Rf
        self.lb = lb
        self.ub = ub
        self.alpha = alpha 
        self.cov = cov
        self.batch_size = batch_size
        self.A = torch.Tensor(A)  
        self.R = torch.zeros((batch_size, P))  
        self.R = np.zeros((batch_size, P))
        self.subnetwork = nn.Sequential(
            nn.Linear(5, 5, bias=False),
            nn.Tanh(),
            nn.Linear(5, 5, bias=False),
            nn.Tanh(),
            nn.Linear(5, 5, bias=False),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        self.R = torch.Tensor(self.R)
        state_variable= x1 
        prev_R = x2      
        for b in range(self.batch_size):
            epsilon = np.random.multivariate_normal(mean=np.zeros(P), cov=self.cov)
            self.R[b, :] = torch.Tensor(self.alpha) + torch.Tensor(A) @ prev_R[b, :] + torch.Tensor(epsilon)
        out = self.R
        weights = self.subnetwork(out)
#         print("weights before rebalancing:", weights.detach().numpy())
        
        output = torch.stack([self.rebalance(batch, self.lb, self.ub) for batch in weights])
#         print('weights after rebalancing',output.detach().numpy())
#         print('self.R', self.R.shape)
#         print('self.Rf',self.Rf.shape)

        updated_wealth = torch.stack([torch.sum(state_variable[i] * output[i] * (1+self.R[i])* self.Rf) for i in range(output.shape[0])])
        updated_return = self.R#         print('returns',self.R+1)
#         print('Rf',self.Rf)
#         print(((output * (self.R + Rf)).sum())
#         print('new portfolio wealth:', updated_wealth.detach().numpy())
        return updated_wealth, updated_return

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
    def __init__(self, P, cov, K, lb, ub, Rf, batch_size, alpha,A):
        super(StackLayer, self).__init__()
        self.blocks = nn.ModuleList([Subnetwork(P, cov, K, lb, ub, Rf, batch_size, alpha,A)])
    
    def forward(self, x1, x2):
        output1 = x1
        output2 = x2 
        for block in self.blocks:
            output1, output2 = block(output1, output2)
        return output1, output2
