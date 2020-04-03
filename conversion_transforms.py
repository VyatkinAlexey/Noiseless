import torch
from torchvision import transforms
import numpy as np


class _ToTensor(object):
    """Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
       to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]."""
    def __init__(self):
        pass
        
    def __call__(self, images):
        image, label = images
        if label is None:
            return transforms.ToTensor()(image)
        return (transforms.ToTensor()(image),
                transforms.ToTensor()(label))

class _ToNumpy(object):
    """Converts torch.Tensor of shape (C x H x W) in the range [0.0, 1.0]
       to numpy.ndarray (H x W x C) in the range [0, 255]."""
    def __init__(self):
        pass
        
    def __call__(self, images):
        image, label = images
        if label is None:
            return (image.permute(1, 2, 0).numpy()*255).astype(np.uint8)
        return ((image.permute(1, 2, 0).numpy()*255).astype(np.uint8),
                (label.permute(1, 2, 0).numpy()*255).astype(np.uint8))
