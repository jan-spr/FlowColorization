import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

class Autoencoder(nn.Module):
    def __init__(self, n_input_channels=4, n_output_channels=2):
        """
        Initializes each part of the convolutional neural network.
        :param n_input_channels: number of input channels
        :param n_output_channels: number of output channels
        """

        super().__init__()

        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.t_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.t_conv1_bn = nn.BatchNorm2d(32)
        self.t_conv2 = nn.ConvTranspose2d(32, n_output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        """
        Implements the forward pass for the given data `x`.
        :param x: The input data.
        :return: The neural network output.
        """
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))

        x = F.relu(self.t_conv1_bn(self.t_conv1(x)))
        x = F.relu(self.t_conv2(x))
        return x