import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseAutoencoder(nn.Module):
    def __init__(self):
        super(ReverseAutoencoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 256, 3)
        self.conv6 = nn.Conv2d(256, 512, 3)
        self.t_conv1 = nn.ConvTranspose2d(512, 256, 3)
        self.t_conv2 = nn.ConvTranspose2d(256, 256, 3)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, 3)
        self.t_conv4 = nn.ConvTranspose2d(128, 128, 3)
        self.t_conv5 = nn.ConvTranspose2d(128, 64, 3)
        self.t_conv6 = nn.ConvTranspose2d(64, 3, 3)

    def forward(self, x):

        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        # Decoder
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.relu(self.t_conv5(x))
        x = torch.sigmoid(self.t_conv6(x))

        return x

class PoolingAutoencoder(nn.Module):
    def __init__(self):
        super(PoolingAutoencoder, self).__init__()
           
        self.conv1 = nn.Conv2d(1, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 64, 3, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(64, 128, 3, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(128, 256, 3, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.t_conv5 = nn.ConvTranspose2d(256, 512, 3, padding=1)
        self.t_conv6 = nn.ConvTranspose2d(512, 3, 3, padding=1)
        
        self.upsampling = nn.Upsample(scale_factor=2)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, x):

        # Encoder
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))

        # Decoder
        x = F.relu(self.t_conv1(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv2(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv3(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv4(x))
        x = self.upsampling(x)
        x = F.relu(self.t_conv5(x))
        x = self.upsampling(x)
        x = torch.sigmoid(self.t_conv6(x))

        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 512, 3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 64, 3, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 128, 3, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.t_conv4 = nn.ConvTranspose2d(128, 256, 3, padding=1)
        self.t_conv5 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.t_conv6 = nn.ConvTranspose2d(256, 512, 3, padding=1)
        self.t_conv7 = nn.ConvTranspose2d(512, 3, 3, padding=1)


    def forward(self, x):

        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        
        # Decoder
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.relu(self.t_conv5(x))
        x = F.relu(self.t_conv6(x))
        x = torch.sigmoid(self.t_conv7(x))

        return x

class DeepAutoencoder(nn.Module):
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 512, 3)
        self.conv2 = nn.Conv2d(512, 256, 3)
        self.conv3 = nn.Conv2d(256, 128, 3)
        self.conv4 = nn.Conv2d(128, 64, 3)
        self.conv5 = nn.Conv2d(64, 32, 3)
        self.conv6 = nn.Conv2d(32, 16, 3)
        self.conv7 = nn.Conv2d(16, 8, 3)
        self.conv8 = nn.Conv2d(8, 4, 3)
        self.conv9 = nn.Conv2d(4, 2, 3)

        self.fc1 = nn.Linear(2*110*110, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 16)

        self.fc7 = nn.Linear(16, 32)
        self.fc8 = nn.Linear(32, 64)
        self.fc9 = nn.Linear(64, 128)
        self.fc10 = nn.Linear(128, 256)
        self.fc11 = nn.Linear(256, 512)
        self.fc12 = nn.Linear(512, 2*110*110)
        
        self.t_conv1 = nn.ConvTranspose2d(2, 4, 3)
        self.t_conv2 = nn.ConvTranspose2d(4, 8, 3)
        self.t_conv3 = nn.ConvTranspose2d(8, 16, 3)
        self.t_conv4 = nn.ConvTranspose2d(16, 32, 3)
        self.t_conv5 = nn.ConvTranspose2d(32, 64, 3)
        self.t_conv6 = nn.ConvTranspose2d(64, 128, 3)
        self.t_conv7 = nn.ConvTranspose2d(128, 256, 3)
        self.t_conv8 = nn.ConvTranspose2d(256, 512, 3)
        self.t_conv9 = nn.ConvTranspose2d(512, 3, 3)


    def forward(self, x):

        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))

        x = x.view(x.size()[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))


        # Decoder
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = F.relu(self.fc12(x))

        x = x.view(x.size()[0], 2, 110,110)
        
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.relu(self.t_conv5(x))
        x = F.relu(self.t_conv6(x))
        x = F.relu(self.t_conv7(x))
        x = F.relu(self.t_conv8(x))
        x = torch.sigmoid(self.t_conv9(x))

        return x


