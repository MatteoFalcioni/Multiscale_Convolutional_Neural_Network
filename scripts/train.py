import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.scnn import SingleScaleCNN
from models.mcnn import MultiScaleCNN
from data.transforms.data_loader import load_las_features  # Replace with actual function once the data is ready
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, LEARNING_RATE_DECAY_EPOCHS, LOSS_FUNCTION, OPTIMIZER, DEVICE, MODEL_SAVE_DIR
from utils.utils import get_model_save_path  # Import the utility function


# Generate synthetic data (replace this with real data loading)
def generate_synthetic_data(num_samples=1000):
    # Generate random 3-channel (e.g., RGB) images of size 128x128 and random labels (0-8 for 9 classes)
    X = torch.randn(num_samples, 3, 128, 128)  # Input images
    y = torch.randint(0, 9, (num_samples,))  # Labels (0-8)
    return X, y


# Prepare DataLoader
def prepare_dataloader(batch_size):
    X, y = generate_synthetic_data()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Train function
def train(model, dataloader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        if isinstance(model, MultiScaleCNN):
            # For MCNN, we need three different inputs (here we use the same synthetic input three times)
            outputs = model(inputs, inputs, inputs)
        else:
            # For SCNN, we use a single input
            outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'Batch [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0


# Function to initialize and train the model
def initialize_and_train():
    # Set device (GPU if available)
    device = DEVICE

    # Choose model: SingleScaleCNN (SCNN) or MultiScaleCNN (MCNN)
    model = MultiScaleCNN().to(device)  # Change to SingleScaleCNN() if needed

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Prepare data loader
    dataloader = prepare_dataloader(BATCH_SIZE)

    # Train the model
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        train(model, dataloader, criterion, optimizer, device)

    # Save the model with dynamic name
    model_name = model.__class__.__name__.lower()  # Get the model class name and make it lowercase
    model_save_path = get_model_save_path(model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

