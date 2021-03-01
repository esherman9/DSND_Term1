''' Procedures for model training, testing, predictions'''

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torchsummary import summary
import datetime
from timeit import default_timer as timer
import pandas as pd
import glob
import os
import utils
from PIL import Image

''' MODEL TRAINING '''
def build_network(architecture, dataloaders, GPU, lr, epochs):

    ''' 1) Initialize pretrained model
        2) Reshape clf to match outputs with no. of classes in the new dataset
        3) Define optimization algorithm for training
        4) Run training'''

    print('Initializing pretrained {} model...'.format(architecture))
    device = 'cuda' if GPU else 'cpu'
    n_classes = 102

    # 1) Initialize pretrained model
    # 2) Reshape clf to match outputs with no. of classes in the new dataset
        # vgg16_bn is ok, but need to update others for proper requires_grad

    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Sequential(
            nn.Linear(4096, n_classes), nn.LogSoftmax(dim=1))
        # Freeze feature detection parameters
        for param in model.features.parameters(): # or model.parameters
            param.requires_grad = False

    elif architecture == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True).to(device)
        # Freeze feature detection parameters
        for param in model.features.parameters(): # or model.features.parameters
            param.requires_grad = False
        model.classifier[-1] = nn.Sequential(
        nn.Linear(4096, n_classes), nn.LogSoftmax(dim=1)).to(device)

    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
        nn.Linear(512, n_classes), nn.LogSoftmax(dim=1))
        # Freeze feature detection parameters
        for param in model.features.parameters(): # or model.parameters
            param.requires_grad = False

    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(1024, n_classes), nn.LogSoftmax(dim=1))
        # Freeze feature detection parameters
        for param in model.features.parameters(): # or model.parameters
            param.requires_grad = False

    elif architecture == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(
            512, n_classes, kernel_size=(1,1), stride=(1,1))
        # Freeze feature detection parameters
        for param in model.features.parameters(): # or model.parameters
            param.requires_grad = False

    summary(model, (3, 224, 224))

    # 3) Define loss function and optimization algorithm for training

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

    # 4) Run training

    print('Beginning {} training...'.format(architecture))
    start = timer()

    model, optimizer, epochs_trained, train_loss, train_acc = train_model(
        model, epochs, criterion, optimizer, device,
        dataloaders['train'], dataloaders['valid'])

    print('Total train time: {:.4f}s'.format(timer() - start))

    # Attach some model attributes
    model.epochs = epochs_trained
    model.optimizer = optimizer
    model.architecture = architecture
    model.cat_to_name = utils.category_names('cat_to_name.json')
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.idx_to_class = {
        idx: class_ for class_, idx in model.class_to_idx.items()
        }

    # Run on testing data (prints loss & acc)
    test_loss, test_acc = testing(
        model, criterion, 'test', dataloaders['test'], device)

    #Summarize results. check this summary is what we want...
    df = pd.DataFrame(
        {'epoch': range(1, epochs_trained+1),
         'train_loss' : train_loss,
         'train_acc' : train_acc,
         'test_loss' : test_loss,
         'test_accuracy' : test_acc
        })
    print(df)

    save_checkpoint(model)

    return model, optimizer


''' MODEL PREDICTION '''
def display_preds(image_path, model, topk, device):

    pd.options.display.float_format = '{:.4f}'.format
    #true_class = image_path.split('\\')[-2] # if sample_image()

    # get predictions
    print(image_path)
    start = timer()
    probs, idxs = predict(Image.open(image_path), model, topk, device)
    print('Prediction time: {:.4f}s'.format(timer() - start))

    df = pd.DataFrame()
    df['flower_idx'] = idxs[0].cpu().numpy()
    df['flower_class'] = df.flower_idx.map(model.idx_to_class)
    df['p'] = probs[0].cpu().numpy()
    df['flower_name'] = df.flower_class.map(model.cat_to_name)
    print(df.head())

def predict(image_path, model, topk, device):
    ''' Predict the class of image '''

    with torch.no_grad():
        model.eval()
        image_tensor = torch.from_numpy(
            utils.process_image(image_path)).float().to(device)

        # insert a singleton "batch" dimension expected by model
        output = model(image_tensor.unsqueeze(0))
        ps = torch.exp(output)
        probs, idxs = ps.topk(topk, dim=1)

    return probs, idxs


''' TRAIN/TEST HELPER FUNCTIONS '''

def train_model(model, epochs, criterion, optimizer, device,
    train_loader, valid_loader):
    '''Train the model, return loss and accuracy'''

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    epochs_trained = 0

    for epoch in range(epochs):
        print('Epoch {}...'.format(epoch+1))
        model.train()
        steps = 0
        validate_every = 50
        running_loss = 0
        running_correct_preds = 0

        for inputs, labels in train_loader:
            steps += 1

            #do training
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() # zero gradients in training
            output = model(inputs)
            loss = criterion(output, labels)

            running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            running_correct_preds += (preds == labels).sum().item()
            loss.backward() # backprop
            optimizer.step() # update weights

            #validate periodically (prints results)
            if steps % validate_every == 0:
                print(f'validation for training batch: {steps}')
                valid_loss, valid_acc = testing(
                    model, criterion, 'valid', valid_loader, device)

        train_loss.append(running_loss / len(train_loader.dataset))
        train_acc.append(running_correct_preds / len(train_loader.dataset))

        # In addition to printing periodic validations during training,
        # capture at the end of each epoch
        validation_loss, validation_acc = testing(
            model, criterion, 'valid', valid_loader, device)
        val_loss.append(validation_loss)
        val_acc.append(validation_acc)

        # Epoch summary
        epochs_trained += 1
        print('-' * 10)
        print(f'Epoch: {epochs_trained} of {epochs} results:')
        print(f'training loss: {train_loss[epoch]:.4f}')
        print(f'training accuracy: {train_acc[epoch]:.4f}')
        print(f'validation loss: {val_loss[epoch]:.4f}')
        print(f'validation accuracy: {val_acc[epoch]:.4f}')
        print('-' * 10)


    return model, optimizer, epochs_trained, train_loss, train_acc

def testing(model, criterion, test_type, dataloader, device):
    '''Runs periodically during training on validation data.
    After training epoch, tests accuracy against the testing set'''

    model.eval()
    running_loss = 0
    running_correct_preds = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)

            running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            running_correct_preds += (preds == labels).sum().item()

        pass_loss = running_loss / len(dataloader.dataset)
        pass_acc = running_correct_preds / len(dataloader.dataset)

        if test_type == 'valid':
            print(f'validation loss: {pass_loss:.4f}, validation acc: {pass_acc:.4f}')
        else:
            print(f'test loss: {pass_loss:.4f}, test acc: {pass_acc:.4f}')

    return pass_loss, pass_acc

def save_checkpoint(model):
    ''' Saves a timestamped checkpoint for trained model '''

    # General points:
    points = {
        'epochs': model.epochs,
        'architecture': model.architecture,
        'class_to_idx' : model.class_to_idx,
        'idx_to_class' : model.idx_to_class,
        'cat_to_name' : model.cat_to_name,
        'model_state_dict': model.state_dict(),
        'optimizer': model.optimizer,
        'optimizer_state_dict': model.optimizer.state_dict()
        }

    # Model-specific points:
    if model.architecture in ['vgg16', 'vgg16_bn']:
        points['classifier'] = model.classifier
    elif model.architecture in ['resnet50']:
        points['fc'] = model.fc

    # Parent folder, model folder, prepend fname with model arch, timestamp
    fpath = 'model_checkpoints\\' \
        + model.architecture + '\\' \
        + model.architecture + '_' \
        + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") \
        + '.tar'

    torch.save(points, fpath)

def load_checkpoint(fpath, device):
    ''' Load model for inference from saved checkpoint '''

    # no checkpoint arg provided, get latest
    if fpath is None:
        fpath = max(glob.iglob('model_checkpoints\\*\\*.tar', recursive=True),
            key=os.path.getctime)

    # Determine architecture from folder name, load model-specific points
    arch = fpath.split('\\')[-2]

    if arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        model.classifier = checkpoint['classifier']

    # Load general points
    # Possible to save/load info about requires_grad?
    model.epochs = checkpoint['epochs']
    model.architecture = checkpoint['architecture']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.cat_to_name = checkpoint['cat_to_name']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    model.to(device)

    summary(model, (3, 224, 224))
    print('Successfully loaded checkpoint from: {}'.format(fpath))

    return model, optimizer
