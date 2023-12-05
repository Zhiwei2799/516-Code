import torch
import numpy as np
from NetWorkGarch import PortfolioModel
from utility import UtilityLoss
from ModelAR_GARCH import Optimize


input1 = torch.ones(1000,1)  # Assuming 10 samples and 5 features for input1
input2 = torch.randn(1000,5, 5)  # Assuming 10 samples, 5 assets, and 5 features for input2

# Model parameters
P = 5  # Number of assets
K = 3  # Number of stack layers
lb = 0.0  # Lower bound for weights
ub = 1.0  # Upper bound for weights
Rf = 1.01  # Risk-free rate
batch_size = 10  # Batch size
alpha=3
omega =3
beta = 3
mu = torch.Tensor((0.05,0.05,0.05,0.05,0.05))

trainer = Optimize(P, K, lb, ub, Rf, batch_size, alpha, omega, beta, mu)
trainer.train(input1, input2, max_epoch,batch_size,save_path)
