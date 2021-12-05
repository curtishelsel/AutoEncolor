from torch.utils.data import DataLoader
from colorizationdataset import ColorizationDataset

def get_dataloader(batch_size, path):

    num_workers = 8 

    data_set = ColorizationDataset(path)

    loader = DataLoader(data_set, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)

    return loader

