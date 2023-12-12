import torch
import numpy as np
from NetWork import PortfolioModel
from utility import UtilityLoss
from Model import Optimize

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
max_epoch = 100
batch_size = 64
x_train = torch.ones((M, 1))  

trainer = Optimize(r, P, cov, Lambda, Delta, K, lb, ub, Rf, batch_size)
trainer.train(x_train, max_epoch, save_path)
x_test = torch.ones((M, 1)) 
batch_size = M
trainer = Optimize(r, P, cov, Lambda, Delta, K, lb, ub, Rf, batch_size)
trainer.test(x_test, save_path)





