import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleScaleCNN(nn.Module):
    """
    A PyTorch implementation of a Single-Scale Convolutional Neural Network (SCNN) for
    image-based tasks. The SCNN is designed to process images at a single scale, learning
    hierarchical features through a series of convolutional, batch normalization, and
    max-pooling layers.

    Architecture Overview:
    - Input: 3-channel image (e.g., RGB) of size 128x128.   Can eventually be changed to fit many channels
    - Four convolutional blocks with Batch Normalization and ReLU activation.
    - Max pooling after the first, second, and fourth convolutional layers.
    - Two fully connected layers for classification at the end.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 64 output channels and a 3x3 kernel.
        bn1 (nn.BatchNorm2d): Batch normalization for the first convolutional layer.
        pool1 (nn.MaxPool2d): Max pooling layer with a 2x2 kernel after the first convolutional block.

        conv2 (nn.Conv2d): Second convolutional layer with 32 output channels and a 3x3 kernel.
        bn2 (nn.BatchNorm2d): Batch normalization for the second convolutional layer.
        pool2 (nn.MaxPool2d): Max pooling layer with a 2x2 kernel after the second convolutional block.

        conv3 (nn.Conv2d): Third convolutional layer with 128 output channels and a 3x3 kernel.
        bn3 (nn.BatchNorm2d): Batch normalization for the third convolutional layer.

        conv4 (nn.Conv2d): Fourth convolutional layer with 128 output channels and a 3x3 kernel.
        bn4 (nn.BatchNorm2d): Batch normalization for the fourth convolutional layer.
        pool4 (nn.MaxPool2d): Max pooling layer with a 2x2 kernel after the fourth convolutional block.

        fc1 (nn.Linear): Fully connected layer that takes the flattened output and produces 256 features.
        fc2 (nn.Linear): Fully connected output layer that produces the final class predictions.
    """

    def __init__(self):
        """
        Initializes the SingleScaleCNN model with its layers.
        """
        super(SingleScaleCNN, self).__init__()

        # First Convolutional Block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second Convolutional Block
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third Convolutional Block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth Convolutional Block
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 16 * 128, 256)  # Adjust input size based on input image size
        self.fc2 = nn.Linear(256, 10)  # Adjust the output size for the number of classes

    def forward(self, x):
        """
        Defines the forward pass of the SingleScaleCNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 128, 128) representing a batch of images.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) representing the class scores.
        """
        # First Convolutional Block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Second Convolutional Block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Third Convolutional Block
        x = F.relu(self.bn3(self.conv3(x)))

        # Fourth Convolutional Block
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Example to test the model
if __name__ == "__main__":
    model = SingleScaleCNN()
    print(model)

    # Dummy input to test the model's output
    input_tensor = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
