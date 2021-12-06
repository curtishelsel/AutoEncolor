# Program trains and makes inference on converting
# grayscale images to RGB image
# Author: Curtis Helsel
# December 2021

import torch
import numpy as np 
from train import *
from models import *
from checkpoint import *
from parser import Argparse
from torch.nn import MSELoss
from torch.optim import Adam
from earlystopping import EarlyStopping
from imageloader import get_dataloader
from colorize import color_image, save_images
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Checks for the available device on
# the system and returns device
def get_device():

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    return device

# Returns the model based on the parameter
# set for the network chosen
def get_model(parameters, device):

    if parameters.network == 'reverse':
        model = ReverseAutoencoder().to(device)
    elif parameters.network == 'pooling':
        model = PoolingAutoencoder().to(device)
    elif parameters.network == 'deep':
        model = DeepAutoencoder().to(device)
    else:
        model = Autoencoder().to(device)

    return model

# Runs the training of the model based on
# the parameters given during program run
# and return trained model to inference
def run_training(parameters, device, model):
    
    # Prints the model setup
    show_model_setup(parameters, device, model)
    
    # Model hyperparameters
    epochs = parameters.epochs
    batch_size = parameters.batch_size 
    learning_rate = parameters.learning_rate
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Scheduler reduces the learning rate on validation loss plateau
    scheduler = ReduceLROnPlateau(optimizer, patience=7, verbose=True)

    # Early stopping stops the training early
    # if not progress of validation loss
    if parameters.early_stopping:
        early_stopping = EarlyStopping()

    # From the program parameters, load the dataset for
    # training the network
    train_path = '../data/train_' + parameters.mode
    train_loader = get_dataloader(batch_size, train_path)

    # From the program parameters, load the dataset for
    # validating the network
    if parameters.validation:
        validation_path = '../data/validation_' + parameters.mode
        validation_loader = get_dataloader(batch_size, validation_path)
    
    # If we are continuing from a previous training, 
    # load the saved checkpoint otherwise set starting epoch
    # and the min loss, otherwise set new start and min loss values
    if parameters.continue_training:
        model, optimizer, start, min_loss = load_checkpoint(model, 
                                            optimizer, parameters)
    else:
        start = 0
        min_loss = np.inf

    # Lists created to use to plot the training
    # and validation loss over epoch length
    train_loss = []
    validation_loss = []
    # Run training for n_epochs specified in config
    for epoch in range(start, epochs):
        print(f'\nEpoch {epoch+1}')

        # Calls the training function in the train module
        loss = train(model, device, train_loader, optimizer, criterion)

        train_loss.append(loss)

        # If validation is selected from the program parameters,
        # run validation function to get the validation loss,
        # and check for decrease to update learning rate
        if parameters.validation:
            loss = validation(model, device, validation_loader, criterion)
            validation_loss.append(loss)
            scheduler.step(loss)

        # Creates and saves the checkpoing of each epoch
        # during training in case of program failure
        checkpoint = create_checkpoint(epoch, loss, model, optimizer)
        save_checkpoint(checkpoint, parameters)
    
        # Saves the best model if the loss is the best so far
        # during training. Works with training and validation loss
        if loss < min_loss:
            save_checkpoint(checkpoint, parameters, save_best=True)
            min_loss = loss

        # If early stopping has been supplied to the program
        # parameters, check if training needs to be stopped
        if parameters.early_stopping:
            early_stopping(loss)
        
            if early_stopping.stop:
                    break

    # Plot the training once it is complete
    plot_training(train_loss, validation_loss, parameters)

    print('Training and evaluation finished')
    
    return model

def main(parameters):

    #  Gets the system device, either CPU or GPU
    device = get_device()

    # Gets the model based on the parameters of the program
    model = get_model(parameters, device)

    # If program is set to train, run the training function
    # otherwise, load the best model for inference
    if parameters.train:
        model = run_training(parameters, device, model)
    else:
        model = load_model(model, parameters, device)
    
    # Colorizes the image using the trained model and save
    # to the source directory
    images = color_image(model, device, parameters.image, parameters.mode)
    save_images(images)

if __name__ == '__main__':

    # Gets the program parameters supplied
    # at program run
    parser = Argparse()
    parameters = parser.args

    main(parameters)
