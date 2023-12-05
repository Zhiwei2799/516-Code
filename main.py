import torch
import numpy as np
from Network import PortfolioModel
from loss import UtilityLoss 

x_train = torch.ones((20000, 1))  

P = 5 # number of stocks
cov = np.array([[0.15, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.15, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.15, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.15, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.15]]) 
Lambda = np.array([0.1, 0.1, 0.2, 0.2, 0.2])
Delta = 1 / 40 
r = 0.03 # return rate
Rf = np.exp(r * Delta) # risk free rate
K = 40 # number of period
M = 20000 # number of path 
lb =0 
ub = 0.5
save_path = 'model_epoch_100.pth'
max_epoch = 10
batch_size = 64
trainer = Optimize(r, P, cov, Lambda, Delta, K, lb, ub, Rf, batch_size)
trainer.train(x_train, max_epoch,batch_size,save_path)
x_test = torch.ones((20000, 1))  

trainer.test(x_test, save_path)





