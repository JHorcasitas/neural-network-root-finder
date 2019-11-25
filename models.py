import torch
import torch.nn as nn


class Pi(nn.Module):
    """PI neural network model as defined in:
    Neural Networks with Problem Decomposition for Finding Real Roots of
    Polynomials"""
    def __init__(self, poly_order: int) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn((1, poly_order)))

    def forward(self, x):
        z = x - self.w
        return torch.prod(z, dim=1, keepdim=True)


class Sigma(nn.Module):
    """Sigma neural network model as defined in:
    Neural Networks with Problem Decomposition for Finding Real Roots of
    Polynomials"""
    def __init__(self, poly_order: int) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn((1, poly_order)))

    def forward(self, x):
        z = torch.abs(x - self.w)
        return torch.sum(torch.log(z), dim=1, keepdim=True)
