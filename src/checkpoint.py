import time
import torch

def create_checkpoint(epoch, loss, model, optimizer):

    checkpoint = {
         'epoch': epoch,
         'loss': loss,
         'state_dict': model.state_dict(),
         'optimizer': optimizer.state_dict()
    }

    return checkpoint

def save_checkpoint(checkpoint):
    
    date_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    checkpoint_path = '../models/'
    checkpoint_path += date_time + '_'
    checkpoint_path += 'epoch_' + str(checkpoint['epoch']) + '_'
    checkpoint_path += 'loss_' + '{:.4f}'.format(checkpoint['loss']) + '_'
    checkpoint_path += 'checkpoint.pt'
    
    torch.save(checkpoint, checkpoint_path)

