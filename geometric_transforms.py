from torchvision import transforms
import numpy as np


class _RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Images randomly with a given probability.
       
       p (float): probability of the images being flipped.
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, images):
        image, label = images
        np.random.seed(0)
        return (transforms.RandomHorizontalFlip(self.p)(image),
                transforms.RandomHorizontalFlip(self.p)(label))

class _RandomVerticalFlip(object):
    """Vertically flip the given PIL Images randomly with a given probability.
    
       p (float): probability of the images being flipped.
    """
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, images):
        image, label = images
        np.random.seed(0)
        return (transforms.RandomVerticalFlip(self.p)(image),
                transforms.RandomVerticalFlip(self.p)(label))
