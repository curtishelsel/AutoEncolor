import torch
import numpy as np
from torchvision import datasets, transforms
from skimage.color import rgb2lab



class ColorizationDataset(datasets.ImageFolder):
    
    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        

        #image_lab = rgb2lab(img)
        #image = image_lab[:,:,0] / 100
        #target = image_lab[:,:,1:] / 128

        self.transforms = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.ToTensor()
                            ])

        image = self.transforms(img)
        #target = self.transforms(target)
        target = transforms.ToTensor()(img)


        return image, target

