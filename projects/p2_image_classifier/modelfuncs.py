''' Procedures for model training, testing, predictions'''

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torchsummary import summary
import datetime

''' MODEL TRAINING '''
def build_network(architecture, dataloaders, GPU, lr, epochs):

    ''' 1) Initialize pretrained model
        2) Reshape classifier to match outputs with no. of classes in the new dataset
        3) Define optimization algorithm for training
        4) Run training'''

    print('Initializing pretrained {} model...'.format(architecture))
    device = 'cuda' if GPU else 'cpu'
    n_classes = 102

    # 1) Initialize pretrained model
    # 2) Reshape classifier to match outputs with no. of classes in the new dataset

    # Ok to add logsoftmax to all?
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = nn.Sequential(
            nn.Linear(4096, n_classes), nn.LogSoftmax(dim=1))

    elif architecture == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        model.classifier[-1] = nn.Sequential(
            nn.Linear(4096, n_classes), nn.LogSoftmax(dim=1))

    elif architecture == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
        nn.Linear(512, n_classes), nn.LogSoftmax(dim=1))

    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(1024, n_classes), nn.LogSoftmax(dim=1))

    elif architecture == 'squeezenet':
        model = models.squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(
            512, n_classes, kernel_size=(1,1), stride=(1,1))

    model.to(device)

    # Freeze feature detection parameters
    # Instead may want to create model, freeze all layers, add new classifiers
        # that way only newly added layers will require gradient
    for param in model.features.parameters():
        param.requires_grad = False

    #summary(model, (3, 224, 224))

    # 3) Define loss function and optimization algorithm for training

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

    # 4) Run training

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []
    epochs_trained = 0

    for epoch in range(epochs):
        train_epoch_loss, train_epoch_acc = train_model(
            model, dataloaders['train'], dataloaders['valid'])
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        # In addition to printing periodic validations during training,
        # capture at the end of each epoch
        validation_loss, validation_acc = validate(model, dataloaders['valid'])
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

    # Add idx_to_class in addition to default class_to_idx for easier lookups
    print(dataloaders['idx_to_class'])
    print(dataloaders['train'].dataset.class_to_idx)

    # get from dataloader itself?:
    model.idx_to_class = dataloaders['idx_to_class_items']

    #Summarize results
    df = pd.DataFrame(
        {'epoch': range(epochs),
         'train_loss' : train_loss,
         'train_acc' : train_acc,
         'validation_loss' : val_loss,
         'validation_accuracy' : val_acc
        })
    print(df)

    # Run on testing data (prints loss & acc)
    test_loss, train_loss = testing(model, dataloaders['train'])

    return model, optimizer, epochs_trained


''' MODEL TESTING '''
def predict(architecture='vgg16_bn', GPU=False):
    ''' 1) Initialize pretrained model
        2) Reshape clf to match outputs to classes in new dataset
        3) Define optimization algorithm for training
        4) Run training '''

    print()


''' TRAIN/TEST HELPER FUNCTIONS '''

def train_model(model, train_loader, valid_loader):
    '''Train the model, return loss and accuracy'''

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
            valid_loss, valid_acc = validate(model, valid_loader)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_correct_preds / len(train_loader.dataset)

    return train_loss, train_acc

def validate(model, dataloader):
    '''Run periodically during training on validation data'''

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

        print(f'validation Loss: {pass_loss:.4f}, validation Acc: {pass_acc:.2f}')

    return pass_loss, pass_acc

def testing(model, dataloader):
    '''After training, test accuracy against the testing set'''

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

        print(f'test loss: {pass_loss:.4f}, test acc: {pass_acc:.4f}')

    return pass_loss, pass_acc

def save_checkpoint(model, criterion, epochs, save_dir):

    points = {'epoch': epoch,
              'class_to_idx' : model.class_to_idx,
              'idx_to_class' : model.idx_to_class,
              'classifier' : model.classifier, # for re-creating our new classifier
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}

    fname = save_dir + datetime.datetime.now()
    print(fname)
    # 2013-04-01T13:01:02
    #torch.save(points, fname)

def load_checkpoint(fpath):
    # return a new model based on saved checkpoint

    model = models.vgg16_bn(pretrained=True)
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)

    checkpoint = torch.load(fpath, map_location=torch.device('cpu'))

    model.epoch = checkpoint['epoch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer
