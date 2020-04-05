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
from utils import slicing, save_result


PATH_TO_DATA = './data'
PATH_TO_DATASET_TABLE = './dataset.csv'
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
    noise_types = NOISE_TYPES
    image_size = IMAGE_SIZE
    frame_size = FRAME_SIZE
    overlay_size = OVERLAY_SIZE
    latent_clean_size = LATENT_CLEAN_SIZE
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    test = TEST
    try:
        opts, args = getopt.getopt(argv, "h", ["train=", "noise_types=",
                                               "image_size=", "frame_size=", "overlay_size=",
                                               "latent_clean_size=",
                                               "batch_size=", "epochs=", "test="])
    except getopt.GetoptError:
        print("./train_test.py --train=<True/False> --noise_types='1, ...' "+
              "[--image_size='width, height' --frame_size='width, width' --overlay_size='width, width' "+
              "--latent_clean_size=<float> "+
              "--batch_size=<int> --epochs=<int>] --test=<True/False>")
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            print("./train_test.py --train=<True/False> --noise_types='1, ...' "+
                  "[--image_size='width, height' --frame_size='width, width' --overlay_size='width, width' "+
                  "--latent_clean_size=<float> "+
                  "--batch_size=<int> --epochs=<int>] --test=<True/False>")
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
    
    total_conv_grads_out_means = []
    total_conv_grads_out_stds = []
    
    total_conv_grads_latent_means = []
    total_conv_grads_latent_stds = []
    
    for epoch in range(epochs):
        model.train()
        print('Epoch {}/{}:'.format(epoch, epochs - 1), flush=True)

        running_loss = 0.0
        running_latent_loss = 0.0

        conv_grads_out = []
        conv_grads_latent = []

        for X_train, y_train in tqdm(data_loader):
            X_train = X_train.to(device)
            # Perform one step of minibatch stochastic gradient descent
            net_latent_first, net_latent_last, net_output = model.forward(X_train)
            loss_out = loss(net_output, X_train)

            optimizer.zero_grad()
            loss_out.backward()
            conv_grads_out.append(model.encoder.complex_conv3.conv[0].weight.grad.detach().cpu().numpy())

        # loop for loss_latent
        for X_train, y_train in tqdm(data_loader):
            X_train = X_train.to(device)
            # Perform one step of minibatch stochastic gradient descent
            net_latent_first, net_latent_last, net_output = model.forward(X_train)

            # now we will create tensor to compare with latent output
            net_latent = torch.cat(tensors=(net_latent_first, net_latent_last), dim=3)
            net_latent_copy = net_latent.detach().clone()
            net_latent_copy = zero_corresp_rows(net_latent_copy, y_train, len(net_latent_first[0, 0, 0, :]))

            loss_latent = latent_loss(net_latent, net_latent_copy)

            optimizer.zero_grad()
            loss_latent.backward()
            conv_grads_latent.append(model.encoder.complex_conv3.conv[0].weight.grad.detach().cpu().numpy())

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
            
        epoch_loss = running_loss / len(data_loader)
        epoch_latent_loss = running_latent_loss / len(data_loader)

        out_grad_mean = np.mean(np.array(conv_grads_out))
        out_grad_std = np.std(np.array(conv_grads_out))

        latent_grad_mean = np.mean(np.array(conv_grads_latent))
        latent_grad_std = np.std(np.array(conv_grads_latent))

        print(f'conv grads out: {out_grad_mean:.6f}, std: {out_grad_std}')
        total_conv_grads_out_means.append(out_grad_mean)
        total_conv_grads_out_stds.append(out_grad_std)
        
        print(f'total_conv_grads_out_means:{total_conv_grads_out_means}') # just to check
        
        print(f'conv grads latent: {latent_grad_mean:.6f},  std: {latent_grad_std:.6f}')
        total_conv_grads_latent_means.append(latent_grad_mean)
        total_conv_grads_latent_stds.append(latent_grad_std)
        
        print('Total Loss: {:.4f}, Latent Loss: {:.4f}'.format(epoch_loss, epoch_latent_loss), flush=True)
    
    
    total_conv_grads_out_means = np.array(total_conv_grads_out_means)
    total_conv_grads_out_stds = np.array(total_conv_grads_out_stds)
    
    total_conv_grads_latent_means = np.array(total_conv_grads_latent_means)
    total_conv_grads_latent_stds = np.array(total_conv_grads_latent_stds)
    
    np.savetxt('total_conv_grads_out_means.csv', [total_conv_grads_out_means], delimiter=',', fmt='%f')
    np.savetxt('total_conv_grads_out_stds.csv', [total_conv_grads_out_stds], delimiter=',', fmt='%f')
    
    np.savetxt('total_conv_grads_latent_means.csv', [total_conv_grads_latent_means], delimiter=',', fmt='%f')
    np.savetxt('total_conv_grads_latent_stds.csv', [total_conv_grads_latent_stds], delimiter=',', fmt='%f')
    
    return model


def test_evaluation(model, dataset,
                    loss, latent_loss,
                    device=DEVICE):
    running_loss = 0.0
    running_latent_loss = 0.0
    for image_number in tqdm(range(len(dataset))):
        path_to_image = dataset.iloc[image_number]['image']
        image = Image.open(path_to_image).convert('L') # PIL Image grayscale
        np_image = np.array(image)[..., np.newaxis] # numpy array from PIL Image
        
        # test loss calculating
        images = torch.stack([ToTensor()(frame) for frame in slicing(np_image, FRAME_SIZE, OVERLAY_SIZE)[0]])
        images = images.to(DEVICE)
        labels = [dataset.iloc[image_number]['label']] * images.shape[0]
        
        model.eval()
        model_latent_first, model_latent_last, model_output = model.forward(images)
        loss_total = loss(model_output, images)

        # now we will create tensor to compare with latent output
        model_latent = torch.cat((model_latent_first, model_latent_last), dim=3)

        model_latent_right = model_latent.detach().clone()
        model_latent_right = zero_corresp_rows(model_latent_right, labels, len(model_latent_first[0, 0, 0, :]))

        # finally combine losses
        loss_latent = latent_loss(model_latent, model_latent_right)
        loss_total += loss_latent

        running_loss += loss_total.item()
        running_latent_loss += loss_latent.item()
    
    test_total_loss = running_loss / len(dataset)
    test_latent_loss = running_latent_loss / len(dataset)
    print('Test Total Loss: {:.4f}, Test Latent Loss: {:.4f}'.format(test_total_loss, test_latent_loss), flush=True)

def test_model(model, dataset, path_to_results):
    running_loss = 0.0
    for image_number in tqdm(range(len(dataset))):
        path_to_image = dataset.iloc[image_number]['image']
        
        image = Image.open(path_to_image).convert('L') # PIL Image grayscale
        np_image = np.array(image)[..., np.newaxis] # numpy array from PIL Image
        
        path_to_save = os.path.join(path_to_results, os.path.basename(path_to_image))
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
        torch.manual_seed(0)
        transforms = [Compose([RandomHorizontalFlip(p=1.0), ToTensor()]),
                      Compose([RandomVerticalFlip(p=1.0), ToTensor()]),
                      Compose([ColorJitter(brightness=(0.9, 2.0), contrast=(0.9, 2.0)), ToTensor()])]

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
        model = AE(1, LATENT_CLEAN_SIZE)
        loss = SSIMLoss()
        latent_loss = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
        model = train_model(model, train_loader,
                            loss, latent_loss,
                            optimizer,
                            epochs=EPOCHS,
                            device=DEVICE)

        # model saving
        path_to_model = './model' + '_{}'.format('_'.join([str(elem) for elem in NOISE_TYPES])) + '.pth'
        torch.save(model, path_to_model)
    
    if TEST:    
        # model loading
        path_to_model = './model' + '_{}'.format('_'.join([str(elem) for elem in NOISE_TYPES])) + '.pth'
        print('{} testing...\n'.format(os.path.basename(path_to_model)))
        model = torch.load(path_to_model)

        dataset=pd.read_csv(PATH_TO_DATASET_TABLE)
        test_dataset = dataset[dataset['phase']=='test']

        # model testing and results saving
        loss = SSIMLoss()
        latent_loss = MSELoss()
        print('{} evaluation on test images'.format(os.path.basename(path_to_model)))
        test_evaluation(model, test_dataset,
                        loss, latent_loss,
                        device=DEVICE)
        print()
        
        path_to_results = PATH_TO_RESULTS + '_{}'.format('_'.join([str(elem) for elem in NOISE_TYPES]))
        if not os.path.exists(path_to_results):
            os.makedirs(path_to_results)
        print('{} running and results saving'.format(os.path.basename(path_to_model)))
        test_model(model, test_dataset, path_to_results)
        
    print('process completed: OK')

if __name__ == "__main__":
    main(sys.argv[1:])