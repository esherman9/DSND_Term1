''' Contains utilities for loading data, processing images'''

import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import helper
import json
import numpy as np
import os

def load_data(data_dir, arch):
    '''load data with torch.dataloader'''

    print('Loading data from {}...'.format(data_dir))
    torch.manual_seed(43)
    batch_size = 24

    # Image transformations
    # Means & stds expected by models trained on ImageNet dataset
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomApply([transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip()], p=0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
        ]),
    'test': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
        ]),
    'valid': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
        ])
    }

    # Create train, test, validation datasets from directories
    image_datasets = {x: datasets.ImageFolder(data_dir + '\\' + x,
        data_transforms[x]) for x in ['train', 'test', 'valid']}

    # Create train, test, validation dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
        batch_size=batch_size, shuffle=True) for x in ['train', 'test', 'valid']}

    return dataloaders

def center_crop(pil, crop_width, crop_height):
    img_width, img_height = pil.size
    return pil.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array '''

    # shrink shorter side to 256, keeping aspect ratio
    thumb_size = (image.width, 256) if image.height < image.width else (256, image.height)
    crop_width, crop_height = (224, 224)
    means, stds = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    image.thumbnail(thumb_size)
    image = center_crop(image, crop_width, crop_height)

    np_image = np.array(image)
    np_image = np_image / 255 # scale values between 0 and 1
    np_image = (np_image - means) / stds # normalize around 0 as expected by network
    np_image = np.transpose(np_image, axes=(2, 0, 1)) # reorder color channel first

    return np_image

def sample_image():
    '''Returns a random image from one of the test folders'''
    n = np.random.choice(range(1, 103))
    path = 'flowers\\test\\' + str(n) + '\\'
    img_path = path + np.random.choice(os.listdir(path))
    #print(img_path)
    return img_path

def category_names(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
