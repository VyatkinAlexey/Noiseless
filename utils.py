import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from PIL import Image
from torchvision.transforms import ToTensor
from conversion_transforms import ToNumpy


def get_image_type(folder_name):
    for i in range(len(folder_name)):
        if folder_name[i] == '_':
            return folder_name[:i]
    return folder_name

def get_noise_type(folder_name):
    for i in reversed(range(len(folder_name))):
        if folder_name[i] == '_':
            return folder_name[i + 1:]

def get_label(folder_name, noise_types):
    """Getting type of image."""
    image_type = get_image_type(folder_name)
    if image_type == 'clean':
        return 'clean_image'
    if image_type == 'noised':
        if get_noise_type(folder_name) not in noise_types:
            return None
        return 'noised_image'
    if image_type == 'only':
        if get_noise_type(folder_name) not in noise_types:
            return None 
        return 'only_noise'

def make_dataset_table(path_to_data, noise_types, path_to_csv_file):
    """Dataset csv table creating."""
    data = np.empty((0, 2))
    for folder_name in os.listdir(path_to_data):
        label = get_label(folder_name, noise_types)
        if label is None:
            continue
        
        path_to_images = os.path.join(path_to_data, folder_name)
        for image_name in os.listdir(path_to_images):
            path_to_image = os.path.join(path_to_data, folder_name, image_name)
            data = np.vstack((data, np.array([path_to_image, label])))

    pd.DataFrame(data, columns=['image', 'label']).to_csv(path_to_csv_file, index=False)
    
def train_test_split(path_to_csv_file, test_size=0.2):
    """Splitting into train and test parts."""
    dataset = pd.read_csv(path_to_csv_file)
    phase = np.empty((0, 1))
    for label in dataset['label'].unique():
        test_number = int(len(dataset[dataset['label']==label]) * test_size) + 1
        train_number = len(dataset[dataset['label']==label]) - test_number
        phase = np.vstack((phase, np.array(['train'] * train_number + ['test'] * test_number).reshape(-1, 1)))
    
    data = np.hstack((np.array(dataset), phase))
    pd.DataFrame(data, columns=['image', 'label', 'phase']).to_csv(path_to_csv_file, index=False)

def get_frame(image, frame_size, overlay_size, index, overlay_mask=None):
    """Getting frame from image (numpy.ndarray) in PIL Image format."""
    height, width, channels = image.shape
    frame_y, frame_x = frame_size
    overlay_y, overlay_x = overlay_size
    
    columns_number = (width - overlay_y) // (frame_y - overlay_y)
    if (width - overlay_y) % (frame_y - overlay_y) != 0:
        columns_number += 1
    
    row = index // columns_number
    column = index % columns_number
    
    end_y = min((column + 1) * frame_y - column * overlay_y, width)
    start_y = end_y - frame_y
    
    end_x = min((row + 1) * frame_x - row * overlay_x, height)
    start_x = end_x - frame_x
    
    if overlay_mask is not None:
        overlay_mask[start_x:end_x, start_y:end_y, :] += np.ones((frame_x, frame_y, channels))
        return image[start_x:end_x, start_y:end_y, :], overlay_mask
    return Image.fromarray(image[start_x:end_x, start_y:end_y, :])
    
def slicing(image, frame_size=(256, 256), overlay_size=(1, 1)):
    """Slicing image (numpy.ndarray) into frames list."""
    height, width, channels = image.shape
    frame_y, frame_x = frame_size
    overlay_y, overlay_x = overlay_size
    
    columns_number = (width - overlay_y) // (frame_y - overlay_y)
    if (width - overlay_y) % (frame_y - overlay_y) != 0:
        columns_number += 1
    
    rows_number = (height - overlay_x) // (frame_x - overlay_x)
    if (height - overlay_x) % (frame_x - overlay_x) != 0:
        rows_number += 1
    
    frames = []
    overlay_mask = np.zeros(image.shape)
    for index in range(columns_number * rows_number):
        frame, overlay_mask = get_frame(image, frame_size, overlay_size, index, overlay_mask)
        frames.append(frame)
    
    return frames, overlay_mask

def gluing(frames, overlay_mask, overlay_size=(1, 1)):
    """Gluing frames list ([numpy.ndarray, ...])
       into one image (numpy.ndarray in the range [0, 255]).
    """
    height, width, _ = overlay_mask.shape
    frame_x, frame_y, _ = frames[0].shape
    overlay_y, overlay_x = overlay_size
    
    image = np.zeros(overlay_mask.shape)
    start_x, start_y = (0, 0)
    end_x, end_y = frame_x, frame_y
    i = 0
    while 1:
        while 1:
            image[start_x:end_x, start_y:end_y, :] += frames[i]
            i += 1
            if end_y == width:
                break
            end_y = min(end_y + frame_y - overlay_y, width)
            start_y = end_y - frame_y
        if end_x == height:
            break
        end_x = min(end_x + frame_x - overlay_x, height)
        start_x = end_x - frame_x
        end_y = frame_y
        start_y = 0
    
    return (image / overlay_mask).astype(np.uint8)

def plot_sliced_image(frames, image_size, frame_size, overlay_size, figsize=(16, 9)):
    """Plotting frames list ([numpy.ndarray, ...])."""
    width, height = image_size
    frame_y, frame_x = frame_size
    overlay_y, overlay_x = overlay_size
    
    columns_number = (width - overlay_y) // (frame_y - overlay_y)
    if (width - overlay_y) % (frame_y - overlay_y) != 0:
        columns_number += 1
    
    rows_number = (height - overlay_x) // (frame_x - overlay_x)
    if (height - overlay_x) % (frame_x - overlay_x) != 0:
        rows_number += 1
    
    fig, axes = plt.subplots(ncols=columns_number, nrows=rows_number, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        im = Image.fromarray(frames[i])
        ax.imshow(im);

def get_predicted_frames(model, frames):
    """Prediction of frames list ([numpy.ndarray, ...]) via model."""
    model.to('cpu')
    model.eval()
    predicted_frames = []
    for frame in frames:
        with torch.set_grad_enabled(False):
            tensor_frame = ToTensor()(np.array(frame)).unsqueeze(0)/1.0
            predicted_frame = model.decoder(torch.cat((model(tensor_frame)[0], 0*model(tensor_frame)[1]), dim=3)).squeeze()
        predicted_frames.append(ToNumpy()(predicted_frame))
    
    return predicted_frames
        
def plot_glued_image(frames, overlay_mask, overlay_size, figsize=(16, 9)):
    """Plotting glued image from frames list ([numpy.ndarray, ...]) and overlay mask."""
    plt.figure(figsize=figsize)
    plt.imshow(gluing(frames, overlay_mask, overlay_size));

def save_result(model, image, frame_size, overlay_size, path_to_save, figsize=(16, 9)):
    frames, overlay_mask = slicing(image, frame_size, overlay_size)
    predicted_frames = get_predicted_frames(model, frames)
    predicted_image = gluing(predicted_frames, overlay_mask, overlay_size)
    
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize)
    axes[0].imshow(Image.fromarray(image))
    axes[1].imshow(Image.fromarray(predicted_image))
    
    fig.savefig(path_to_save)