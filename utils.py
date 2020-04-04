import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns

from PIL import Image
from conversion_transforms import _ToTensor, _ToNumpy


def get_image_and_label(path_to_images):
    paths_to_images = os.listdir(path_to_images)
    return paths_to_images[-1], paths_to_images[0]

def make_dataset_table(path_to_data, path_to_csv_file):
    """Dataset csv table creating."""
    data = np.empty((0, 2))
    for folder_name in os.listdir(path_to_data):
        path_to_images = os.path.join(path_to_data, folder_name)
        image_name, label_name = get_image_and_label(path_to_images)
        
        path_to_image = os.path.join(path_to_data, folder_name, image_name)
        path_to_label = os.path.join(path_to_data, folder_name, label_name)

        data = np.vstack((data, np.array([path_to_image, path_to_label])))

    pd.DataFrame(data, columns=['image', 'label']).to_csv(path_to_csv_file, index=False)
    
def train_test_split(path_to_csv_file, test_size=0.2):
    """Splitting into train and test parts."""
    dataset = pd.read_csv(path_to_csv_file)
    
    test_size = int(len(dataset) * test_size) + 1
    train_size = len(dataset) - test_size
    phase = np.array(['train'] * train_size + ['test'] * test_size)
    
    data = np.hstack((np.array(dataset), phase.reshape(-1, 1)))
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
    predicted_frames = []
    for frame in frames:
        model.eval()
        with torch.set_grad_enabled(False):
            predicted_frame = model.to('cpu')(_ToTensor()((np.array(frame), None)).unsqueeze(0)/1.0).squeeze()
        predicted_frames.append(_ToNumpy()((predicted_frame, None)))
    
    return predicted_frames
        
def plot_glued_image(frames, overlay_mask, overlay_size, figsize=(16, 9)):
    """Plotting glued image from frames list ([numpy.ndarray, ...]) and overlay mask."""
    plt.figure(figsize=figsize)
    plt.imshow(gluing(frames, overlay_mask, overlay_size));


def CountPlot(latent_vector, eps=1e-3):
	"""Plot histogram of zero/non-zero elements for latent vector"""
    binary_vector = np.array(["Non-zero" if abs(value) > eps else "Close to zero" for value in latent_vector])
    binary_df = pd.DataFrame(columns=["zero_prox"], data=binary_vector)
    ax = sns.countplot(x="zero_prox", data=binary_df)
    ax.set(xlabel="latent vector components values")