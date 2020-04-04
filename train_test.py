import sys, getopt
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, ToTensor
from torch.utils.data import ConcatDataset, DataLoader

from tqdm import tqdm

from utils import make_dataset_table, train_test_split
from dataset import DenoisingDataset
from loss import SSIMLoss, MSELoss
from model import AE
from utils import save_result


PATH_TO_DATA = './data'
PATH_TO_DATASET_TABLE = './dataset.csv'
PATH_TO_MODEL = './model.pth'
PATH_TO_RESULTS = './results'

TRAIN = False

NOISE_TYPES = []
# IMAGE_SIZE = (3840, 2160) # (width, height) Ultra HD 4K
# IMAGE_SIZE = (1920, 1080) # (width, height) Full HD
IMAGE_SIZE = (1080, 720) # (width, height) HD
FRAME_SIZE = (256, 256) # (frame_width, frame_height)
OVERLAY_SIZE = (5, 5) # (stride_y, stride_x)
LATENT_CLEAN_SIZE = 0.9
BATCH_SIZE = 4
EPOCHS = 40

TEST = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def arguments_parsing(argv):
    train = TRAIN
    test = TEST
    noise_types = NOISE_TYPES
    image_size = IMAGE_SIZE
    frame_size = FRAME_SIZE
    overlay_size = OVERLAY_SIZE
    latent_clean_size = LATENT_CLEAN_SIZE
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    try:
        opts, args = getopt.getopt(argv, "h", ["train=", "noise_types=",
                                               "image_size=", "frame_size=", "overlay_size=",
                                               "latent_clean_size=",
                                               "batch_size=", "epochs=", "test="])
    except getopt.GetoptError:
        print("./train_test.py --train=<True/False> [--noise_types='1, ...' "+
              "--image_size='width, height' --frame_size='width, height' --overlay_size='width, height'] "+
              "--latent_clean_size=<float> "+
              "--batch_size=<int> --epochs=<int> --test=<True/False>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("./train_test.py --train=<True/False> [--noise_types='1, ...' "+
                  "--image_size='width, height' --frame_size='width, height' --overlay_size='width, height'] "+
                  "--latent_clean_size=<float> "+
                  "--batch_size=<int> --epochs=<int> --test=<True/False>")
            sys.exit()
        elif opt == "--train":
            if arg == 'False':
                train = False
            elif arg == 'True':
                train = True
        elif opt == "--noise_types":
            noise_types = arg.split(", ")
        elif opt == "--image_size":
            image_size = tuple(map(int, arg.split(", ")))
        elif opt == "--frame_size":
            frame_size = tuple(map(int, arg.split(", ")))
        elif opt == "--overlay_size":
            overlay_size = tuple(map(int, arg.split(", ")))
        elif opt == "--latent_clean_size":
            latent_clean_size = float(arg)
        elif opt == "--batch_size":
            batch_size = int(arg)
        elif opt == "--epochs":
            epochs = int(arg)
        elif opt == "--test":
            if arg == 'False':
                test = False
            elif arg == 'True':
                test = True
            
    return train, noise_types, image_size, frame_size, overlay_size, latent_clean_size, batch_size, epochs, test

def zero_corresp_rows(latent_vector, labels, num_latent_clean):
    """
    inp_tensor is torch.tensor of shape (batch_size, latent_dim)
    y_array is array of shape (batch_size,)
    """
    for index, label in enumerate(labels):
        if label == 'clean_image': # we should zero last components
            latent_vector[index, ..., num_latent_clean:] = 0 # in torch this works
        elif label == 'only_noise': # we should zero first components
            latent_vector[index, ..., :num_latent_clean] = 0 # in torch this works

    return latent_vector

def train_model(model, data_loader,
                loss, latent_loss,
                optimizer,
                epochs=EPOCHS,
                device=DEVICE):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        print('Epoch {}/{}:'.format(epoch, epochs - 1), flush=True)

        running_loss = 0.0
        running_latent_loss = 0.0
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            
            # Perform one step of minibatch stochastic gradient descent
            optimizer.zero_grad()
            model_latent_first, model_latent_last, model_output = model.forward(images)
            loss_total = loss(model_output, images)

            # now we will create tensor to compare with latent output
            model_latent = torch.cat((model_latent_first, model_latent_last), dim=3)
            
            model_latent_right = model_latent.detach().clone()
            model_latent_right = zero_corresp_rows(model_latent_right, labels, len(model_latent_first[0, 0, 0, :]))

            # finally combine losses
            loss_latent = latent_loss(model_latent, model_latent_right)
            loss_total += loss_latent

            loss_total.backward()
            optimizer.step()
            
            running_loss += loss_total.item()
            running_latent_loss += loss_latent.item()
            
#             # Perform one step of minibatch stochastic gradient descent
#             optimizer.zero_grad()
#             model_latent_first, model_latent_last, model_output = model.forward(images)
#             loss_total = loss(model_output, images)
            
#             # loss calculating
#             if labels[0] == 'clean_image':
#                 loss_latent = latent_loss(model_latent_last, 
#                                           torch.zeros(size=model_latent_last.size(), device=device))
#                 loss_total += loss_latent
#             elif labels[0] == 'noised_image':
#                 pass
#             elif labels[0] == 'only_noise':
#                 loss_latent = latent_loss(model_latent_first, 
#                                           torch.zeros(size=model_latent_first.size(), device=device))
#                 loss_total += loss_latent
 
#             loss_total.backward()
#             optimizer.step()
        
#             running_loss += loss_total.item()
            
        epoch_loss = running_loss / len(data_loader)
        epoch_latent_loss = running_latent_loss / len(data_loader)
        print('Total Loss: {:.4f}, Latent Loss: {:.4f}'.format(epoch_loss, epoch_latent_loss), flush=True)
        
    return model

def test_model(model, dataset):
    for image_number in tqdm(range(len(dataset))):
        path_to_image = dataset.iloc[image_number]['image']
        image = Image.open(path_to_image) # PIL Image
        np_image = np.array(image) # numpy array from PIL Image
        
        path_to_save = os.path.join(PATH_TO_RESULTS, os.path.basename(path_to_image))
        save_result(model, np_image, FRAME_SIZE, OVERLAY_SIZE, path_to_save, figsize=(16, 9))

def main(argv):
    TRAIN, NOISE_TYPES, IMAGE_SIZE, FRAME_SIZE, OVERLAY_SIZE, LATENT_CLEAN_SIZE, BATCH_SIZE, EPOCHS, TEST = arguments_parsing(argv)
    
    if TRAIN:
        print('model training with parameters:\n'+
              'noise types = {}\n'.format(NOISE_TYPES)+
              'image size = {}\n'.format(IMAGE_SIZE)+
              'frame size = {}\n'.format(FRAME_SIZE)+
              'overlay size = {}\n'.format(OVERLAY_SIZE)+
              'latent clean size = {}\n'.format(LATENT_CLEAN_SIZE)+
              'batch size = {}\n'.format(BATCH_SIZE)+
              'number of epochs = {}\n'.format(EPOCHS))
        
        # dataset table creating
        make_dataset_table(PATH_TO_DATA, NOISE_TYPES, PATH_TO_DATASET_TABLE)
        train_test_split(PATH_TO_DATASET_TABLE, test_size=0.2)

        # dataset and dataloader creating
        np.random.seed(0)
        transforms = [Compose([RandomHorizontalFlip(p=1.0), ToTensor()]),
                      Compose([RandomVerticalFlip(p=1.0), ToTensor()]),
                      Compose([ColorJitter(hue=0.5), ToTensor()])]

        train_dataset = []
        for transform in transforms:
            dataset = DenoisingDataset(dataset=pd.read_csv(PATH_TO_DATASET_TABLE),
                                       image_size=IMAGE_SIZE,
                                       frame_size=FRAME_SIZE,
                                       overlay_size=OVERLAY_SIZE,
                                       phase='train',
                                       transform=transform)
            train_dataset = ConcatDataset([train_dataset, dataset])

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, # can be set to True only for train loader
                                  num_workers=0)

        # model training
        model = AE(3, 3, LATENT_CLEAN_SIZE)
        loss = SSIMLoss()
        latent_loss = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        model = train_model(model, train_loader,
                            loss, latent_loss,
                            optimizer,
                            epochs=EPOCHS,
                            device=DEVICE)

        # model saving
        torch.save(model, PATH_TO_MODEL)
    
    if TEST:
        print('model testing...')
        
        # model loading
        model = torch.load(PATH_TO_MODEL)

        dataset=pd.read_csv(PATH_TO_DATASET_TABLE)
        test_dataset = dataset[dataset['phase']=='test']

        # model testing and results saving
        if not os.path.exists(PATH_TO_RESULTS):
            os.makedirs(PATH_TO_RESULTS)
        test_model(model, test_dataset)
        
    print('process completed: OK')
#     os.system("pause")

if __name__ == "__main__":
    main(sys.argv[1:])
