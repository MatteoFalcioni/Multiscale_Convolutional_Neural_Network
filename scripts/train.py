import torch
from utils.utils import save_model


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
        # For MCNN, we need three different scaled input images
        # Training the MCNN automatically trains the SCNN because of the architecture
        outputs = model(inputs, inputs, inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'Batch [{i + 1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set and returns the average loss.
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass for MCNN
            outputs = model(inputs, inputs, inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(dataloader)


def train_epochs(model, dataloader, criterion, optimizer, scheduler, epochs, device, save_dir):
    """
    Trains the model over multiple epochs and applies learning rate decay. It saves the trained model
    after the training process completes.

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

    # Save the model
    save_model(model, save_dir)
