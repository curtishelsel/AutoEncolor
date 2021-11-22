# Author: Curtis Helsel

import sys
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from autoencoder import Autoencoder
from tqdm import tqdm

def get_device():
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    return device

def get_dataloader(batch_size):

    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(
            "../data/train",
            transforms.Compose([
                    #transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
            ]))
    test_set = datasets.ImageFolder(
            "../data/test",
            transforms.Compose([
                    transforms.ToTensor(),
            ]))

    train_loader = DataLoader(train_set, batch_size=batch_size, 
                                shuffle=True,num_workers=4)
    test_loader = DataLoader(test_set, batch_size=10, 
                                shuffle=True,num_workers=4)

    return train_loader, test_loader

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    
    data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1)
    ])
    
    # Iterate over entire training samples (1 epoch)
    with tqdm(train_loader, unit='batch') as tepoch:
        for index, batch in enumerate(tepoch):

            tepoch.set_description(f"Epoch {epoch+1}")

            target_data, _ = batch
            input_data = data_transforms(target_data)
            
            
            # Push data/label to correct device
            input_data = input_data.to(device)
            
            # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
            optimizer.zero_grad()
            
            # Do forward pass for current set of data
            output = model(input_data)
            
            # Compute loss based on criterion
            loss = criterion(output, target_data)
            
            # Computes gradient based on final loss
            loss.backward()
            
            # Store loss
            losses.append(loss.item())
            
            # Optimize model parameters based on learning rate and gradient 
            optimizer.step()
    
        
    train_loss = float(np.mean(losses))
    print('Train set: Average loss: {:.4f}'.format(train_loss))

    return train_loss

def get_model_output(model, device, test_loader):
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    data = None
    output = None
    # Set torch.no_grad() to disable gradient computation and backpropagation
    data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1)
    ])
    with torch.no_grad():
        for sample in test_loader:

            data, target = sample
            
            input_data = data_transforms(data)
            input_data = input_data.to(device)
            
            output = model(input_data)
            break
    
    return data, output

def save_images(images, name):
    
    for index, image in enumerate(images):
        plt.subplot(5, 2, index + 1)
        plt.imshow(np.transpose(image.numpy(), (1,2,0)))
        plt.axis('off')
        if index == 9:
            break
        
    plt.savefig(name)
    
if __name__ == '__main__':
    
    device = get_device()
    model = Autoencoder().to(device)
    summary(model, (1,64,64))

    # Model hyperparameters
    epochs = 75 
    batch_size = 100
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Get data from ImageNet dataset and put into dataloader
    train_loader, test_loader = get_dataloader(batch_size)

    # Run training for n_epochs specified in config 
    trian_loss = 0.0
    for epoch in range(epochs):

        train_loss = train(model, device, train_loader,
                            optimizer, criterion, epoch, batch_size)

    # Get the predicted output from test data and save images to file
    data, output = get_model_output(model, device, train_loader)
    
    save_images(data, 'data.jpg')
    save_images(output, 'output.jpg')

    print("Loss is {:2.2f} after {} epochs".format(train_loss, epochs))
    print("Training and evaluation finished")
