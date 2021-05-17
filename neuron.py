import torch
import torch.nn as nn


class SoftLIF(nn.Module):
    def __init__(self, tau_ref, tau_rc, v_th, gamma):
        super(SoftLIF, self).__init__()
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc
        self.v_th = v_th
        self.gamma = gamma

    def forward(self, x):
        x = 1+torch.exp(torch.true_divide(x, self.gamma))
        x = self.gamma*torch.log(x)
        x = 1+torch.true_divide(self.v_th, x)
        x = self.tau_ref+self.tau_rc*torch.log(x)
        x = torch.true_divide(1, x)
        return x

