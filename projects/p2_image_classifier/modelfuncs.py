''' Procedures for model training and testing'''

import torch
from torchvision import datasets, transforms, models
from torchsummary import summary

''' MODEL TRAINING '''
def build_network(architecture, dataloaders, GPU=False, lr=0.01, epochs=3):

''' 1) Initialize pretrained model
    2) Reshape classifier to match outputs with no. of classes in the new dataset
    3) Define optimization algorithm for training
    4) Run training '''

# 1) Initialize pretrained model

    print('Initializing pretrained {} model...'.format(architecture))
    device = 'cuda' if GPU else 'cpu'

    # Select pretrained model based on arch args
    # Way to use the arg 'architecture' directly like models.arch()...?
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True).to(device)
    elif architecture == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True).to(device)
    elif architecture == 'vgg13':
        model = models.vgg16_bn(pretrained=True).to(device)
    elif architecture == 'vgg13_bn':
        model = models.vgg16_bn(pretrained=True).to(device)

# 2) Reshape classifier to match outputs with no. of classes in the new dataset

    model.classifier[-1].out_features = 102

    # add LogSoftmax to get true probabilities from output
    model.classifier.extend(LogSoftmax(dim=1))
    #summary(model, (3, 224, 224))

# 3) Define loss fucntion and optimization algorithm for training

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)

# 4) Run training

    # Freeze feature detection parameters
    for param in model.features.parameters():
        param.requires_grad = False

    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for epoch in range(epochs):
        train_epoch_loss, train_epoch_acc =
            train_model(model, dataloaders['train'], dataloaders['valid'])
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        # In addition to printing periodic validations during training,
        # capture at the end of each epoch
        validation_loss, validation_acc = validate(model, dataloaders['valid'])
        val_loss.append(validation_loss)
        val_acc.append(validation_acc)

        # Epoch summary
        print('-' * 10)
        print(f'Epoch: {epoch+1} of {epochs} results:')
        print(f'training loss: {train_loss[epoch]:.4f}')
        print(f'training accuracy: {train_acc[epoch]:.4f}')
        print(f'validation loss: {val_loss[epoch]:.4f}')
        print(f'validation accuracy: {val_acc[epoch]:.4f}')
        print('-' * 10)

    # Add idx_to_class in addition to default class_to_idx for easier lookups
    training_data_c2i = datasets.ImageFolder(train_dir, train_data_transforms)
    model.idx_to_class = {
        idx: class_ for class_, idx in training_data.class_to_idx.items()
        }

''' MODEL TESTING '''
def predict(architecture='vgg16_bn', GPU=False):
''' 1) Initialize pretrained model
    2) Reshape classifier to match outputs with no. of classes in the new dataset
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

def save_checkpoint(model, criterion):
    # called from build_network
    print()
