import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleCNN(nn.Module):
    """
    A PyTorch implementation of a Single-Scale Convolutional Neural Network (SCNN) for
    image-based tasks. The SCNN is designed to process images at a single scale, learning
    hierarchical features through a series of convolutional, batch normalization, and
    max-pooling layers.

    """

    def __init__(self):
        """
        Initializes the SingleScaleCNN model with its layers.
        """
        super(SingleScaleCNN, self).__init__()

        # First Convolutional Block: 3x128x128 --> 32x64x64  (recall: in Torch, channel dimension goes first)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  # Output: 32x128x128
        self.bn1 = nn.BatchNorm2d(32)   # Batch normalization doesn't change the spatial dimension
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 32x64x64 (MaxPool changes spatial dimension)

        # Second Convolutional Block: 32x64x64 --> 64x32x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Output: 64x64x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x32x32

        # Third Convolutional Block: 64x32x32 --> 128x16x16
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Output: 128x32x32
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x16x16

        # Fourth Convolutional Block: 128x16x16 unchanged here (no max pooling)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # Output: 128x16x16
        self.bn4 = nn.BatchNorm2d(128)  # Output: 128x16x16

        # Fifth and final Convolutional Block: 128x16x16 --> 128x8x8
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  # Output: 128x16x16
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x8x8

