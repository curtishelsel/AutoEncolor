# This file contains the function to load
# a dataset into the data loader
# Author: Curtis Helsel
# December 2021

from torch.utils.data import DataLoader
from colorizationdataset import ColorizationDataset

# Sets the number of workers for the system and
# load a dataset into the Dataloader object 
def get_dataloader(batch_size, path):

    num_workers = 8 

    data_set = ColorizationDataset(path)

    loader = DataLoader(data_set, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)

    return loader

