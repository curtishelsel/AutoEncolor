# Author: Curtis Helsel

import torch
import numpy as np 
import matplotlib.pyplot as plt
from train import *
from checkpoint import *
from parser import Argparse
from torch.nn import MSELoss
from torch.optim import Adam
from models import Autoencoder, PoolingAutoencoder, ReverseAutoencoder
from earlystopping import EarlyStopping
from imageloader import get_dataloader
from colorize import color_image, save_images
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_device():
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    
    return device

def get_model(parameters, device):

    if parameters.network == 'reverse':
        model = ReverseAutoencoder().to(device)
    elif parameters.network == 'pooling':
        model = PoolingAutoencoder().to(device)
    else:
        model = Autoencoder().to(device)

    return model


def run_training(parameters, device, model):
    
    show_model_setup(parameters, device, model)

    # Model hyperparameters
    epochs = parameters.epochs
    batch_size = parameters.batch_size 
    learning_rate = parameters.learning_rate
    criterion = MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Scheduler reduces the learning rate on validation loss plateau
    scheduler = ReduceLROnPlateau(optimizer, patience=7, verbose=True)

    # Early stopping stops the training early if not progress of validation loss
    if parameters.early_stopping:
        early_stopping = EarlyStopping()

    train_path = '../data/train_' + parameters.mode
    train_loader = get_dataloader(batch_size, train_path)

    if parameters.validation:
        validation_path = '../data/validation_' + parameters.mode
        validation_loader = get_dataloader(batch_size, validation_path)
    
    # If we are continuing from a previous training, 
    # load the saved checkpoint otherwise set starting epoch
    # and the min loss
    if parameters.continue_training:
        model, optimizer, start, min_loss = load_checkpoint(model, 
                                            optimizer, parameters)
    else:
        start = 0
        min_loss = np.inf

    train_loss = []
    validation_loss = []
    # Run training for n_epochs specified in config
    for epoch in range(start, epochs):
        print(f'\nEpoch {epoch+1}')

        loss = train(model, device, train_loader, optimizer, criterion)

        train_loss.append(loss)

        if parameters.validation:
            loss = validation(model, device, validation_loader, criterion)
            validation_loss.append(loss)
            scheduler.step(loss)

        checkpoint = create_checkpoint(epoch, loss, model, optimizer)
        save_checkpoint(checkpoint, parameters)
    
        if loss < min_loss:
            save_checkpoint(checkpoint, parameters, save_best=True)
            min_loss = loss

        if parameters.early_stopping:
            early_stopping(loss)
        
            if early_stopping.stop:
                    break

    plot_training(train_loss, validation_loss, parameters)

    print("Training and evaluation finished")

    return model

def main(parameters):

    device = get_device()

    model = get_model(parameters, device)

    if parameters.train:
        model = run_training(parameters, device, model)
    else:
        model = load_model(model, parameters)
    
    images = color_image(model, device, parameters.image, parameters.mode)
    save_images(images)

if __name__ == '__main__':

    parser = Argparse()
    parameters = parser.args

    main(parameters)
