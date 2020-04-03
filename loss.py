import torch
from pytorch_msssim import SSIM


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, predictions, labels):
        return self.mse_loss(predictions.view(-1), labels.view(-1))

class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3, nonnegative_ssim=True)
        
    def forward(self, predictions, labels):
        return 1 - self.ssim_module(predictions, labels)
