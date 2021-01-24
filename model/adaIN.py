import torch
from torch import nn


class AdaIN(nn.Module):
    def __init__(self, eps=1e-8):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        mu_x = torch.mean(x, dim=(2, 3), keepdim=True)
        mu_y = torch.mean(y, dim=(2, 3), keepdim=True)
        var_x = torch.var(x, dim=(2, 3), keepdim=True) + self.eps
        var_y = torch.var(y, dim=(2, 3), keepdim=True) + self.eps
        return var_y.sqrt() * (x - mu_x) / var_x.sqrt() + mu_y
