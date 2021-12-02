import os
import time
import torch

def create_checkpoint(epoch, loss, model, optimizer):

    checkpoint = {
         'epoch': epoch,
         'loss': loss,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict()
    }

    return checkpoint

def save_checkpoint(checkpoint, save_best=False):
    
    current_path = '../models/current_checkpoint.pt'
    torch.save(checkpoint, current_path)

    if save_best:
        best_path = '../models/best_model.pt'
        torch.save(checkpoint, best_path)
        print('New best model saved to ' + best_path)


def load_checkpoint(model, optimizer, checkpoint_path=None):

    if checkpoint_path == None:
        checkpoint_path = '../models/current_checkpoint.pt'

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']

    return model, optimizer, epoch, loss
