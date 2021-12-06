# This file contains the functions to train
# a model, validation for the model and
# plot graphs for loss visualization
# Author: Curtis Helsel
# December 2021

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import no_grad
from torchsummary import summary

# Trains the network model
def train(model, device, train_loader, optimizer, criterion):

    # Set model to train mode before each epoch
    model.train()

    train_loss = 0.0

    # Iterate over entire training samples (1 epoch)
    with tqdm(train_loader, unit='batch') as tepoch:
        for index, batch in enumerate(tepoch):

            tepoch.set_description('  Training')

            input_data, target_data = batch

            # Push data/label to correct device
            input_data = input_data.to(device)
            target_data = target_data.to(device)

            # Reset optimizer gradients Avoids grad accumulation
            optimizer.zero_grad()

            # Do forward pass for current set of data
            output = model(input_data)

            # Compute loss based on criterion
            loss = criterion(output, target_data)

            # Computes gradient based on final loss
            loss.backward()

            # Optimize model parameters based on learning rate and gradient
            optimizer.step()
            
            # Update the running training loss
            loss_difference = loss.data - train_loss
        
            train_loss += ((1 / (index + 1)) * loss_difference) 

    print('Train Loss: {:.4f}'.format(train_loss))

    return float(train_loss)

# Validates the model with a validation set not seen by the model
# during training
def validation(model, device, validation_loader, criterion):

    model.eval()

    validation_loss = 0.0

    # Iterate over entire training samples (1 epoch)
    with tqdm(validation_loader, unit='batch') as tepoch:
        with no_grad():
            for index, batch in enumerate(tepoch):
                
                tepoch.set_description('Validation')

                input_data, target_data = batch

                # Push data/label to correct device
                input_data = input_data.to(device)
                target_data = target_data.to(device)

                # Do forward pass for current set of data
                output = model(input_data)

                # Compute loss based on criterion
                loss = criterion(output, target_data)
                
                # Update the running training loss
                loss_difference = loss.data - validation_loss

                validation_loss += ((1 / (index + 1)) * loss_difference) 

    print('Validation Loss: {:.4f}'.format(validation_loss))

    return float(validation_loss)

# This function prints the model set up for
# a given model provided by the program input parameters
def show_model_setup(parameters, device, model):

    print('Epochs: {}'.format(parameters.epochs))
    print('Batch Size: {}'.format(parameters.batch_size))
    print('Validation: {}'.format(parameters.validation))
    print('Training Set: {}'.format(parameters.mode))
    print('Learning Rate: {}'.format(parameters.learning_rate))
    print('Early Stopping: {}'.format(parameters.early_stopping))
    print('Torch device selected: {}'.format(device))

    summary(model, (1,128,128))

# Plots a visualization of the network losses
def plot_training(train_loss, validation_loss, parameters):

    epochs = parameters.epochs
    mode = parameters.mode
    
    title = 'Network Loss\n' + str(epochs) + ' Epochs on '
    title += mode.capitalize() + ' Dataset'

    path = '../figures/' + mode + str(epochs) + 'epochs'

    if parameters.early_stopping:
        path += 'earlystopping'

    path += '.png'

    plt.plot(train_loss, label='Train Loss')
    if parameters.validation:
        plt.plot(validation_loss, label='Validation Loss')

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend(loc='upper right')
    
    # Either shows or saves graph based on
    # parameter provided at program run
    if parameters.show_plot:
        plt.show()
    else:
        plt.savefig(path)

