import sys, getopt
import os
import torch
torch.manual_seed(0)

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm


NOISE_TYPE = 18
NOISE_LEVEL = 4

def arguments_parsing(argv):
    path_to_clean_images = ""
    path_to_corrupted_images = ""
    noise_type = NOISE_TYPE
    try:
        opts, args = getopt.getopt(argv, "h", ["path_to_clean_images=", "path_to_corrupted_images=",
                                               "noise_type="])
    except getopt.GetoptError:
        print("./gaussian_noise_overlay.py --path_to_clean_images=<str> "+
              "--path_to_corrupted_images=<str> "+
              "[--noise_type=<int>]")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("./gaussian_noise_overlay.py --path_to_clean_images=<str> "+
                  "--path_to_corrupted_images=<str> "+
                  "[--noise_type=<int>]")
            sys.exit()
        elif opt == "--path_to_clean_images":
            path_to_clean_images = os.path.join(arg)
        elif opt == "--path_to_corrupted_images":
            path_to_corrupted_images = os.path.join(arg)
        elif opt == "--noise_type":
            noise_type = int(arg)
    
    return path_to_clean_images, path_to_corrupted_images, noise_type

class _GaussianNoise(object):
    """Adding Gaussian Noise to the image.
       
       mean (float): mean of Gaussian distribution
       std (float): std of Gaussian distribution
    """
    def __init__(self, mean=0.1, std=0.1):
        self.mean = mean
        self.std = std
              
    def __call__(self, image):
        tensor = ToTensor()(image)
        return ToPILImage()(torch.clamp(tensor + torch.randn(tensor.size()) * self.std + self.mean, 0.0, 1.0))
    
def noise_overlay(path_to_clean_images, path_to_corrupted_images, noise_type, noise_level):
    images_names = os.listdir(path_to_clean_images)
    for image_name in tqdm(images_names):
        path_to_image = os.path.join(path_to_clean_images, image_name)
        image = _GaussianNoise()(Image.open(path_to_image))
        
        path_to_image = os.path.join(path_to_corrupted_images, image_name)
        ext = os.path.splitext(path_to_image)[1]
        path_to_save = os.path.join(os.path.splitext(path_to_image)[0] +  
                                    '_{}_{}'.format(noise_type, noise_level) + ext)
        image.save(path_to_save)

def main(argv):
    path_to_clean_images, path_to_corrupted_images, noise_type = arguments_parsing(argv)
    noise_overlay(path_to_clean_images, path_to_corrupted_images, noise_type, NOISE_LEVEL)

if __name__ == "__main__":
    main(sys.argv[1:])
