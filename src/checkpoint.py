# This file contains the functions for creating, saving
# loading checkpoints and models
# Author: Curtis Helsel
# December 2021

import torch
import sys

# Creates a training checkpoint dictionary with
# number of epochs, current loss, model state, 
# and optimizer state dictionaries
def create_checkpoint(epoch, loss, model, optimizer):

    checkpoint = {
         'epoch': epoch,
         'loss': loss,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict()
    }

    return checkpoint

# Saves the checkpoint provided with path set by
# paramaters of the network. If the validation loss
# is the best so far in the network, the best path is
# saved to the models directory.
def save_checkpoint(checkpoint, parameters, save_best=False):
    
    path = '../models/current_' + parameters.network + '_checkpoint.pt'
    torch.save(checkpoint, path)

    if save_best:
        best_path = '../models/best_' + parameters.network + '_model.pt'
        torch.save(checkpoint, best_path)
        print('New best model saved to ' + best_path)

# Loads the checkpoint from the models directory
# and applies the states of the optimiser and model
# of the last saved checkpoint
def load_checkpoint(model, optimizer, parameters):

    path = '../models/current_' + parameters.network + '_checkpoint.pt'

    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

    except Exception as e:
        print(e)
        print('Unable to continue training.')
        print('You have not saved any {} checkpoints.'.format(parameters.network))
        print('Train a model first using -t without -c flag.')
        sys.exit(1)

    return model, optimizer, epoch, loss

# Loads the model of the most recently saved
# best model based on the network parameters given
def load_model(model, parameters, device):

    
    best_path = '../models/best_' + parameters.network + '_model.pt'

    try:
        best_model = torch.load(best_path, map_location=device)
        model.load_state_dict(best_model['model_state_dict'])

    except Exception as e:
        print(e)
        print('You have not saved any best {} models.'.format(parameters.network))
        print('Train a model first using --train or -t')
        sys.exit(1)

    return model
