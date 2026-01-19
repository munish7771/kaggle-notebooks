import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Encoder, self).__init__()
        # Input: 3 x 32 x 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32 x 32 x 32
        self.pool1 = nn.MaxPool2d(2, 2)                         # 32 x 16 x 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 x 16 x 16
        self.pool2 = nn.MaxPool2d(2, 2)                         # 64 x 8 x 8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 128 x 8 x 8
        self.pool3 = nn.MaxPool2d(2, 2)                         # 128 x 4 x 4
        
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.uppool1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1) # 64 x 8 x 8
        
        self.uppool2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)  # 32 x 16 x 16
        
        self.uppool3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1)   # 3 x 32 x 32

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 4, 4)
        
        x = self.uppool1(x)
        x = F.relu(self.conv1(x))
        
        x = self.uppool2(x)
        x = F.relu(self.conv2(x))
        
        x = self.uppool3(x)
        x = torch.sigmoid(self.conv3(x)) # Pixels effectively 0-1 if standardized properly, usually Tanh or Sigmoid depending on norm
        # Note: Our transforms normalize to approx -1..1 range (mean 0.5, std 0.5 roughly). 
        # So we should probably output raw logits or Tanh? 
        # Or simplistic approach: just linear output and MSE loss.
        # Let's stick to Linear output for now to match the normalized target range distribution roughly.
        return x 

class DAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(DAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        # We need decoder to be somewhat symmetric or capable of reconstructing
        # Slightly adjusting decoder logic to match encoder properly
        self.decoder_input_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 64x8x8
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 32x16x16
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2)   # 3x32x32

    def forward(self, x):
        z = self.encoder(x)
        
        out = F.relu(self.decoder_input_fc(z))
        out = out.view(out.size(0), 128, 4, 4)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = self.deconv3(out) 
        # Output is logits/unbounded to be compared via MSE with normalized images
        return out

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        # Re-use the encoder architecture logic for fairness
        self.encoder = Encoder(latent_dim=256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x) # Encoder output is linear, add activation if needed before classifier
        x = self.classifier(x)
        return x

class FineTunedClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super(FineTunedClassifier, self).__init__()
        self.encoder = encoder
        # Assume encoder latent dim is 256 based on DAE default
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # We might want to freeze encoder or not, controlled by optimizer usually
        x = self.encoder(x)
        x = F.relu(x)
        x = self.classifier(x)
        return x
