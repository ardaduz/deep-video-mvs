import math

import torch

from dvmvs.utils import freeze_batchnorm


class GPlayer(torch.nn.Module):
    def __init__(self, device):
        super(GPlayer, self).__init__()
        self.gamma2 = torch.nn.Parameter(torch.randn(1).to(device).float(), requires_grad=True)
        self.ell = torch.nn.Parameter(torch.randn(1).to(device).float(), requires_grad=True)
        self.sigma2 = torch.nn.Parameter(torch.randn(1).to(device).float(), requires_grad=True)
        self.device = device

    def forward(self, D, Y):
        """
        :param D: Distance matrix
        :param Y: Stacked outputs from encoder
        :return: Z: transformed latent space
        """
        # Support for these operations on Half precision is low at the moment, handle everything in Float precision
        batch, latents, channel, height, width = Y.size()
        Y = Y.view(batch, latents, -1).float()
        D = D.to(self.device).float()

        # MATERN CLASS OF COVARIANCE FUNCTION
        # ell > 0, gamma2 > 0, sigma2 > 0 : EXPONENTIATE THEM !!!
        K = torch.exp(self.gamma2) * (1 + math.sqrt(3) * D / torch.exp(self.ell)) * torch.exp(-math.sqrt(3) * D / torch.exp(self.ell))
        I = torch.eye(latents, device=self.device, dtype=torch.float32).expand(batch, latents, latents)
        C = K + torch.exp(self.sigma2) * I
        Cinv = C.inverse()
        Z = K.bmm(Cinv).bmm(Y)
        Z = torch.nn.functional.relu(Z)
        return Z

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(GPlayer, self).train(mode)
        self.apply(freeze_batchnorm)
