import sys
import argparse
import utils
import modelfuncs
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
# from torchvision import datasets, transforms, models
from torchsummary import summary
from collections import OrderedDict
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import helper
import os
from PIL import Image

# torch.manual_seed(43) # does it make sense to set this for entire project?

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="root directory for images")
parser.add_argument("save_dir", type=str, help="directory to save model checkpoints")
parser.add_argument("arch", type=str, help="model architecture (e.g. 'vgg16')")
parser.add_argument("--learn_rate", type=str, help="learning rate hyperparameter")
parser.add_argument("--epochs", type=str, help="number of epochs to train over")
parser.add_argument("--GPU", help="use GPU for training", action="store_true")

# hidden units doesn't seem like something we want to parameterize...
# classifiers can have many hidden layers, would need to make sure everything aligns with selected model
# parser.add_argument("--hidden_units", type=int, help="learning rate hyperparameter")

args = parser.parse_args()

# validate architecture
accepted_archs = ['vgg13', 'vgg16', 'vgg16_bn']
if args.arch.lower() not in accepted_archs:
    sys.exit('Invalid archicture')

if args.GPU:
    print("GPU in use")

print('Initiating training for {} model...'.format(args.arch))

dataloaders = utils.load_data(args.data_dir, args.arch)
cat_names = utils.category_names('cat_to_name.json') # add as arg?

# builds network, trains, displays training stats, saves checkpoint
modelfuncs.build_network(args.arch, args.GPU)
