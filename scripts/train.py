import torch
from models.scnn import SingleScaleCNN
from models.mcnn import MultiScaleCNN
from utils.utils import get_model_save_path  # Import the utility function


def train(model, dataloader, criterion, optimizer, device):
    """
    Trains the given model for one epoch on the provided DataLoader. It performs a forward pass, computes
    the loss, performs backpropagation, and updates the model weights.

    Args:
    - model (nn.Module): The PyTorch model to be trained.
    - dataloader (DataLoader): DataLoader object containing the training data.
    - criterion (nn.Module): The loss function to optimize.
    - optimizer (optim.Optimizer): The optimizer to use for training.
    - device (torch.device): The device (CPU or GPU) to perform computations on.
    """
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


def train_epochs(model, dataloader, criterion, optimizer, scheduler, epochs, device, save_dir):
    """
    Trains the model for a specified number of epochs.

    Args:
    - model (nn.Module): The PyTorch model to be trained.
    - dataloader (DataLoader): DataLoader object for the training data.
    - criterion (nn.Module): The loss function to optimize.
    - optimizer (optim.Optimizer): The optimizer to use for training.
    - scheduler (optim.lr_scheduler): Learning rate scheduler for adjusting the learning rate.
    - epochs (int): Number of epochs to train the model.
    - device (torch.device): The device (CPU or GPU) to perform computations on.
    - save_dir (str): Directory to save the trained model.
    """
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training step
        train(model, dataloader, criterion, optimizer, device)

        # Step the scheduler to adjust the learning rate
        scheduler.step()

        print(f"Learning Rate after Epoch {epoch + 1}: {scheduler.get_last_lr()}")

    # Save the model with dynamic name
    model_name = model.__class__.__name__.lower()
    model_save_path = get_model_save_path(model_name, save_dir)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
