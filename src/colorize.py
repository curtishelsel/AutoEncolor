# This file contains functions to color an image
# and save images to source directory
# Author: Curtis Helsel
# December 2021

import os
import numpy as np
from torch import no_grad
from imageloader import get_dataloader
from PIL import Image
from torchvision import transforms

# This function colorizes and image either
# from the path provided or a random image
# from the test image set
def color_image(model, device, path, mode):

    # Transform the image to grayscale for
    # model inference
    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()])

    # If a path has not been supplied, pull
    # a random image from the test directory
    if path == None:
        path = '../data/test_' + mode + '/faces/'
        path += np.random.choice(os.listdir(path))
        
    # Using PIL, open the image and resize
    # to 128x128 for use in the model inference
    original_image = Image.open(path)

    original_image.thumbnail((128, 128))
    
    # Transform the image to grayscale and
    # set the batch size to 1 for pytorch
    # model inference
    input_image = transform(original_image)

    input_image = input_image.unsqueeze(0)

    # Set model to eval mode to notify all layers.
    model.eval()

    # Make the inference (colorize image)
    with no_grad():

        input_image = input_image.to(device)

        colorized_image = model(input_image)
    
    # Create a dictionary of the original image,
    # network input image, and network output 
    # (colorized)image
    images = {
        'original'  : original_image, 
        'input'     : input_image,
        'colorized' : colorized_image}

    return images

# This function saves the set of images
# to the source directory after resizing to 256x 256
def save_images(images):

    for image in images:

        # If the image is not a PIL image, convert to PIL image
        # to be able to save
        if image != 'original':
            pil_image = transforms.ToPILImage()(images[image].squeeze(0))
            images[image] = pil_image 

        # Resize the image for better visualization
        images[image] = images[image].resize((256,256), Image.ANTIALIAS)
                
        # Save PIL image to directory
        images[image].save(image + '.jpg')
