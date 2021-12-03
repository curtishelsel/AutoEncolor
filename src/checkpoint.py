import os
import time
import torch
import sys

def create_checkpoint(epoch, loss, model, optimizer):

    checkpoint = {
         'epoch': epoch,
         'loss': loss,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict()
    }

    return checkpoint

def save_checkpoint(checkpoint, parameters, save_best=False):
    
    path = '../models/current_' + parameters.network + '_checkpoint.pt'
    torch.save(checkpoint, path)

    if save_best:
        best_path = '../models/best_' + parameters.network + '_model.pt'
        torch.save(checkpoint, best_path)
        print('New best model saved to ' + best_path)


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

def load_model(model, parameters):

    best_path = '../models/best_' + parameters.network + '_model.pt'
    try:
        best_model = torch.load(best_path)
        model.load_state_dict(best_model['model_state_dict'])
    except Exception as e:
        print(e)
        print('You have not saved any best {} models.'.format(parameters.network))
        print('Train a model first using --train or -t')
        sys.exit(1)

    return model
