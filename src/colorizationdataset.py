# This file contains the class for the 
# custom dataset for loading images to use
# in training and with DataLoader
# Author: Curtis Helsel
# December 2021

from torchvision import datasets, transforms

# This class loads images in a subfolder,
# and converts them from RGB to grayscale
class ColorizationDataset(datasets.ImageFolder):
    
    # Gets individual image and transforms
    # image to grayscale, returns original
    # image as target and grayscale input
    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        # Transforms the image to grayscale
        # and make it a tensor
        self.transforms = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.ToTensor()
                            ])

        image = self.transforms(img)
        target = transforms.ToTensor()(img)

        return image, target

