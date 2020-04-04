import torch
from pytorch_msssim import ssim


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, predictions, labels):
        return self.mse_loss(predictions, labels)

class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        
    def forward(self, predictions, labels):
        ssim_loss = 1 - ssim(predictions, labels, data_range=1.0, size_average=True, nonnegative_ssim=True)
        return ssim_loss
