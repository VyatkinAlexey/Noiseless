import torch
import numpy as np

class ToNumpy(object):
    """Converts torch.Tensor of shape (C x H x W) in the range [0.0, 1.0]
       to numpy.ndarray (H x W x C) in the range [0, 255]."""
    def __init__(self):
        pass
        
    def __call__(self, image):
        return (image.permute(1, 2, 0).numpy()*255).astype(np.uint8)
