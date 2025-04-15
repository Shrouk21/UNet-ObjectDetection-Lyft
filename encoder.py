import numpy as np
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_channels, n_filters=32, dropout_rate=0, maxpooling=True):
        super(Encoder, self).__init__()
        self.input_channels = input_channels
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.maxpooling = maxpooling


        self.conv1 = nn.Conv2d(input_channels, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        skip_connection = x # Store for the contractive input
        if self.maxpooling:
            x = self.maxpool(x)
        return x , skip_connection

