class UtilityLoss(torch.nn.Module):
    def __init__(self):
        super(UtilityLoss, self).__init__()

    def forward(self, W, gamma):
        objective = - (W - gamma/2)**2
        return -objective.mean()
