import os
import numpy as np
from torch import no_grad
from imageloader import get_dataloader
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def color_image(model, device, path, mode):

    transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()])

    if path == None:
        path = '../data/train_' + mode + '/faces/'
        path += np.random.choice(os.listdir(path))
        
    original_image = Image.open(path)

    original_image.thumbnail((128, 128))

    input_image = transform(original_image)

    input_image = input_image.unsqueeze(0)

    # Set model to eval mode to notify all layers.
    model.eval()

    with no_grad():

        input_image = input_image.to(device)

        colorized_image = model(input_image)

    images = {
        'original'  : original_image, 
        'input'     : input_image,
        'colorized' : colorized_image}

    return images


def save_images(images):

    for image in images:

        if image != 'original':
            pil_image = transforms.ToPILImage()(images[image].squeeze(0))
            images[image] = pil_image 


        images[image] = images[image].resize((256,256), Image.ANTIALIAS)
                
        images[image].save(image + '.jpg')
