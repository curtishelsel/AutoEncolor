import numpy as np
import torch

class EarlyStopping:

    def __init__(self, patience=15, threshold=1e-4):

        self.patience = patience
        self.epoch_count = 0
        self.best_loss = np.Inf 
        self.threshold = threshold
        self.stop = False

    def __call__(self, current_loss):
    
        loss_threshold = float(self.best_loss) * (1 - self.threshold)

        if current_loss < loss_threshold:
            self.epoch_count = 0
            self.best_loss = current_loss
        else:
            self.epoch_count += 1

        if self.epoch_count > self.patience:
            self.stop = True
            print('Maximum epoch threshold exeeded')
            print('Training stopping early')


        





