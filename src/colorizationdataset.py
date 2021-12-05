from torchvision import datasets, transforms

class ColorizationDataset(datasets.ImageFolder):
    
    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        self.transforms = transforms.Compose([
                            transforms.Grayscale(),
                            transforms.ToTensor()
                            ])

        image = self.transforms(img)
        target = transforms.ToTensor()(img)

        return image, target

