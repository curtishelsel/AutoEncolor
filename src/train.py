from tqdm import tqdm
import numpy as np

def train(model, device, train_loader, optimizer, criterion):

    # Set model to train mode before each epoch
    model.train()

    train_loss = 0.0

    # Iterate over entire training samples (1 epoch)
    with tqdm(train_loader, unit='batch') as tepoch:
        for index, batch in enumerate(tepoch):

            tepoch.set_description('  Training')

            input_data, target_data = batch

            #input_data = input_data.float()
            #target_data = target_data.float()

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

            # Optimize model parameters based on learning rate and gradient
            optimizer.step()
        
            train_loss += ((1 / (index + 1)) * (loss.data - train_loss)) 

    return train_loss

def validation(model, device, validation_loader, criterion):

    model.eval()

    validation_loss = 0.0

    with tqdm(validation_loader, unit='batch') as tepoch:
        for index, batch in enumerate(tepoch):
            
            tepoch.set_description('Validation')

            input_data, target_data = batch

            #input_data = input_data.float()
            #target_data = target_data.float()

            # Push data/label to correct device
            input_data = input_data.to(device)
            target_data = target_data.to(device)

            # Do forward pass for current set of data
            output = model(input_data)

            # Compute loss based on criterion
            loss = criterion(output, target_data)

            validation_loss += ((1 / (index + 1)) * (loss.data - validation_loss)) 

    return validation_loss
