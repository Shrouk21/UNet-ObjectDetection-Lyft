import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, expansive_input_channels, contractive_input_channels, n_filters=32):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(expansive_input_channels, n_filters, kernel_size=2, stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_filters + contractive_input_channels, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.up(x)  # Upsample input
        x = torch.cat((x, skip), dim=1)  # Concatenate with skip connection along channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
