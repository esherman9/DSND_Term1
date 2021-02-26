''' Contains utilities for loading data, processing images'''

import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import helper
import json

def load_data(data_dir, arch):
    '''load data with torch.dataloader'''

    print('Loading data from {}...'.format(data_dir))

    torch.manual_seed(43)

    # train_dir = data_dir + '/train'
    # valid_dir = data_dir + '/valid'
    # test_dir = data_dir + '/test'
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

    # Create train, test, validation datasets
    image_datasets = {x: datasets.ImageFolder(data_dir + '\\' + x,
        data_transforms[x]) for x in ['train', 'test', 'valid']}

    # Create train, test, validation dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
        batch_size=batch_size, shuffle=True) for x in ['train', 'test', 'valid']}

    # Load the datasets with ImageFolder
    # training_data = datasets.ImageFolder(train_dir, train_data_transforms)
    # test_data = datasets.ImageFolder(test_dir, test_data_transforms)
    # validation_data = datasets.ImageFolder(valid_dir, valid_data_transforms)

    # Store idx_to_class in addition to default class_to_idx for easier lookups
    # Added to model after training in modelfuncs
    # idx_to_class_items = training_data.class_to_idx.items()

    # Define dataloaders from the image datasets and the trainforms
    # for d_set in [training_data, test_data, validation_data]:
    #     train_dataloader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
    #     test_dataloader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)
    #     valid_dataloader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True)

    dataloaders['idx_to_class'] = image_datasets['train'].class_to_idx.items()
    #
    # dataloaders = {
    #     'train': train_dataloader,
    #     'test': test_dataloader,
    #     'valid': valid_dataloader,
    #     'idx_to_class_items': idx_to_class_items
    #     }

    return dataloaders

def category_names(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
