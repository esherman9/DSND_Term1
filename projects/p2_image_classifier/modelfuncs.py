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
        # Freeze feature detection parameters only
        # Replace last layer of classifier or fc layer to fit flower data
    # 3) Define optimization
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True).to(device)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Sequential(
            nn.Linear(4096, n_classes), nn.LogSoftmax(dim=1)).to(device)
        optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

    elif architecture == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True).to(device)
        # print(model.epochs)
        # exit()
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[-1] = nn.Sequential(
            nn.Linear(4096, n_classes), nn.LogSoftmax(dim=1)).to(device)
        optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=True).to(device)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(2048, n_classes), nn.LogSoftmax(dim=1)).to(device)
        optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)

    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True).to(device)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(1024, n_classes), nn.LogSoftmax(dim=1)).to(device)
        optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

    # Having trouble training squeezenet, need to look into this more.
    # Removed from choices
    elif architecture == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True).to(device)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Conv2d(
            512, n_classes, kernel_size=(1,1), stride=(1,1)).to(device)
        optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

    # appears to be a bug in torchsummary with densenet
    # https://github.com/sksq96/pytorch-summary/issues/2
    if architecture != 'densenet121':
        summary(model, (3, 224, 224))

    # Define loss function
    criterion = nn.NLLLoss()

    # 4) Run training for n epochs
    print('Beginning {} training...'.format(architecture))
    start = timer()
    model, optimizer, epochs_trained, \
    train_loss, train_acc, test_loss, test_acc = train_model(
        model, epochs, criterion, optimizer, device, dataloaders)
    print('Total train time: {:.4f}s'.format(timer() - start))

    # Attach some model attributes
    model.epochs = epochs_trained
    model.optimizer = optimizer
    model.architecture = architecture
    model.cat_to_name = utils.category_names('cat_to_name.json')
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx
    model.idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}

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

    return model


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

def train_model(model, epochs, criterion, optimizer, device, dataloaders):
    '''Train the model, return loss and accuracy'''

    train_loss, train_acc = [], []
    test_loss, test_acc = [], []

    try:
        epochs_trained = model.epochs # not yet assigned for newly created model
    except AttributeError:
        epochs_trained = 0

    for epoch in range(epochs):
        print('Epoch {}...'.format(epoch+1))
        model.train()
        steps = 0
        validate_every = 50
        running_loss = 0
        running_correct_preds = 0

        for inputs, labels in dataloaders['train']:
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
                    model, criterion, 'valid', dataloaders['valid'], device)

        train_loss.append(running_loss / len(dataloaders['train'].dataset))
        train_acc.append(running_correct_preds / len(dataloaders['train'].dataset))

        # In addition to printing periodic validations during training,
        # test at the end of each epoch
        testing_loss, testing_acc = testing(
            model, criterion, 'test', dataloaders['test'], device)
        test_loss.append(testing_loss)
        test_acc.append(testing_acc)

        # Epoch summary
        epochs_trained += 1
        print('-' * 10)
        print(f'Epoch: {epochs_trained} of {epochs} results:')
        print(f'training loss: {train_loss[epoch]:.4f}')
        print(f'training accuracy: {train_acc[epoch]:.4f}')
        print(f'testing loss: {test_loss[epoch]:.4f}')
        print(f'testing accuracy: {test_acc[epoch]:.4f}')
        print('-' * 10)

    return model, optimizer, epochs_trained, train_loss, train_acc, test_loss, test_acc

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
    if model.architecture in ['vgg16', 'vgg16_bn', 'densenet121','squeezenet']:
        points['classifier'] = model.classifier
    elif model.architecture in ['resnet50']:
        points['fc'] = model.fc

    parent_dir = 'model_checkpoints'
    model_dir = model.architecture
    fdir = os.path.join(parent_dir, model_dir)
    fname = model.architecture + '_' \
        + datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") \
        + '.tar'
    fpath = os.path.join(fdir, fname)

    if not os.path.exists(fdir):
        os.makedirs(fdir, exist_ok=True)

    torch.save(points, fpath)

def load_checkpoint(fname, device):
    ''' Load model for inference from saved checkpoint '''

    # add better validation... in argparse?
    # no checkpoint arg provided, get latest
    fpath = None

    if fname is None:
        fpath = max(glob.iglob('model_checkpoints\\*\\*.tar', recursive=True),
            key=os.path.getctime)

    # user provided a path or wrong file type
    elif len(fname.split('\\')) > 1:
        print('Invalid path. Enter file name only in model_checkpoints\\arch\\')
        exit()

    elif fname.endswith('.tar') == False:
        print("Invalid file type. Checkpoint should be '.tar'")
        exit()

    # file name seems valid, search
    else:
        for root, dirs, files in os.walk('model_checkpoints', topdown=False):
            for fil in files:
                if fil == fname:
                    fpath = os.path.join(root, fil)

    if fpath is None:
        print("Invalid. Enter a file name in model_checkpoints\\[arch]\\[model].tar")
        exit()
    # Determine architecture from folder name, load model-specific points
    arch = fpath.split('\\')[-2]

    if arch =='vgg16':
        model = models.vgg16(pretrained=True)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        model.classifier = checkpoint['classifier']
    elif arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        model.classifier = checkpoint['classifier']
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        model.fc = checkpoint['fc']
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        model.classifier = checkpoint['classifier']
    elif arch == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True)
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

    print('Successfully loaded checkpoint from: {}'.format(fpath))

    return model, optimizer
