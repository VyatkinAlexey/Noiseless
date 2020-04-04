import sys, getopt
import os
import torch
torch.manual_seed(0)

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, RandomErasing
from tqdm import tqdm


NOISE_TYPE = 19
NOISE_LEVEL = 4

def arguments_parsing(argv):
    path_to_clean_images = ""
    path_to_corrupted_images = ""
    noise_type = NOISE_TYPE
    try:
        opts, args = getopt.getopt(argv, "h", ["path_to_clean_images=", "path_to_corrupted_images=",
                                               "noise_type="])
    except getopt.GetoptError:
        print("./erasing_noise_overlay.py --path_to_clean_images=<str> "+
              "--path_to_corrupted_images=<str> "+
              "[--noise_type=<int>]")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("./erasing_noise_overlay.py --path_to_clean_images=<str> "+
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

class _Random_Erasing(object):
    """Randomly selects a rectangle region in an image and erases its pixels.
       
       p: probability that the random erasing operation will be performed.
       scale: range of proportion of erased area against input image.
       ratio: range of aspect ratio of erased area.
       value: erasing value. Default is 0. If a single int, it is used to erase all pixels.
              If a tuple of length 3, it is used to erase R, G, B channels respectively.
              If a str of 'random', erasing each pixel with random values.
       inplace: boolean to make this transform inplace. Default set to False.
    """
    def __init__(self, p=1, scale=(0., 0.7), ratio=(0.7, 0.5), value='random', inplace=False):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value 
        self.inplace = inplace
              
    def __call__(self, image):
        tensor = ToTensor()(image)
        return ToPILImage()(RandomErasing(self.p, self.scale, self.ratio, self.value, self.inplace)(tensor))
    
def noise_overlay(path_to_clean_images, path_to_corrupted_images, noise_type, noise_level):
    paths_to_images = os.listdir(path_to_clean_images)
    for path_to_image in paths_to_images:
        image = _Random_Erasing()(Image.open(path_to_image))
        
        ext = os.path.splitext(path_to_corrupted_images)[1]
        path_to_save = os.path.join(os.path.splitext(path_to_corrupted_images)[0],
                                    '_{}_{}'.format(noise_type, noise_level) + ext)
        image.save(path_to_save)

def main(argv):
    path_to_clean_images, path_to_corrupted_images, noise_type = arguments_parsing(argv)
    noise_overlay(path_to_corrupted, path_to_reference, path_for_processed, noise_type, NOISE_LEVEL)

if __name__ == "__main__":
    main(sys.argv[1:])
