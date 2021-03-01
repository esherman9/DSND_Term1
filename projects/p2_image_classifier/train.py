import sys
import argparse
import utils
import modelfuncs
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
from collections import OrderedDict
import os


# torch.manual_seed(43) # does it make sense to set this for entire project?
print("PyTorch Version: ",torch.__version__)

parser = argparse.ArgumentParser(
    'train selected pytorch model to detect flower type from image')

parser.add_argument(
    'data_dir', type=str, help='root directory for images')
# parser.add_argument( # Prevent user error in save directories
#     'save_dir', type=str, help='directory to save model checkpoints')
parser.add_argument(
    'arch', type=str, help="model architecture (e.g. 'vgg16')",
    choices = ['vgg16', 'vgg16_bn', 'resnet18', 'densenet121', 'squeezenet'])
parser.add_argument(
    '--learn_rate', type=int, default=0.01, help='learning rate hyperparameter')
parser.add_argument(
    '--epochs', type=int, default=2, help='number of epochs to train over')
parser.add_argument(
    '--GPU', help='use GPU for training', action='store_true')

args = parser.parse_args()

if args.GPU:
    print('GPU in use')

dataloaders = utils.load_data(args.data_dir, args.arch)
cat_names = utils.category_names('cat_to_name.json')

# builds network, trains, displays training stats, saves checkpoint
model = modelfuncs.build_network(
    args.arch, dataloaders, args.GPU, args.learn_rate, args.epochs)
