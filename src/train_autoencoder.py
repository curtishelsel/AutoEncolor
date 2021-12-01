# Author: Curtis Helsel

import sys
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from autoencoder import Autoencoder
from colorizationdataset import ColorizationDataset
from tqdm import tqdm
from skimage.color import lab2rgb

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

    image_count = 1000
    data_set = ColorizationDataset("../data/train")

    #train_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 2, replace=False))
    data_set = torch.utils.data.Subset(data_set, np.random.choice(len(data_set), image_count, replace=False))

    train_count = int(0.7 * image_count)
    test_count = int(0.3 * image_count)
    
    train_set, test_set = random_split(data_set, (train_count, test_count))

    train_loader = DataLoader(train_set, batch_size=batch_size, 
                                shuffle=True,num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, 
                                shuffle=True,num_workers=4)

    return train_loader, train_loader

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    
    
    # Iterate over entire training samples (1 epoch)
    with tqdm(train_loader, unit='batch') as tepoch:
        for index, batch in enumerate(tepoch):

            tepoch.set_description(f"Epoch {epoch+1}")

            input_data, target_data = batch
            
            input_data = input_data.float()
            target_data = target_data.float()

            # Push data/label to correct device
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            
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
    
    target = None
    output = None
    batch_size = 0
    channels = 3
    lightness = 100
    ab_color = 128

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for sample in test_loader:

            input_data, target = sample

            input_data = input_data.to(device)
            
            output = model(input_data)
            break
    
    return target, output #image_input, image_output

def save_images(images, name):
    
    for index, image in enumerate(images):
        plt.subplot(1, 1, index + 1)
        image = np.transpose(image.numpy(), (1,2,0))
        plt.imshow(image)
        plt.axis('off')
        if index == 0:
            break
        
    plt.savefig(name)
    
if __name__ == '__main__':
    
    device = get_device()
    model = Autoencoder().to(device)
    #model = torch.load("../models/model")
    summary(model, (1,128,128))

    # Model hyperparameters
    epochs = 30
    batch_size = 100 
    learning_rate = 0.001
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Get data from ImageNet dataset and put into dataloader
    train_loader, test_loader = get_dataloader(batch_size)

    for x in train_loader:
        continue

    
    # Run training for n_epochs specified in config 
    trian_loss = 0.0
    for epoch in range(epochs):

        train_loss = train(model, device, train_loader,
                           optimizer, criterion, epoch, batch_size)

    # Get the predicted output from test data and save images to file
    data, output = get_model_output(model, device, train_loader)
    
    save_images(data, 'data.jpg')
    save_images(output, 'output.jpg')
    torch.save(model, "../models/model")

    #print("Loss is {:2.2f} after {} epochs".format(train_loss, epochs))
    print("Training and evaluation finished")
