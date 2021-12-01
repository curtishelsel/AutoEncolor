
def create_checkpoint(epoch, loss, model, optimizer):

    checkpoint = {
         'epoch': epoch,
         'loss': loss,
         'state_dict': model.state_dict(),
         'optimizer': optimizer.state_dict()
    }

    return checkpoint
                                                                 }
