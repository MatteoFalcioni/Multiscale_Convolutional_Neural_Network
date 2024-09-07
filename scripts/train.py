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


def train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device, save_dir):
    """
    Trains the MCNN model over multiple epochs with early stopping and applies learning rate decay.
    After each epoch, the model is evaluated on a validation set, and training stops if the validation
    loss does not improve for a specified number of epochs (patience). Saves the trained model after
    the training process completes.

    Args:
    - model (nn.Module): The MCNN model to be trained.
    - train_loader (DataLoader): DataLoader object for the training data.
    - val_loader (DataLoader): DataLoader object for the validation data.
    - criterion (nn.Module): The loss function to optimize.
    - optimizer (optim.Optimizer): The optimizer to use for training.
    - scheduler (optim.lr_scheduler): Learning rate scheduler for adjusting the learning rate.
    - epochs (int): Number of epochs to train the model.
    - device (torch.device): The device (CPU or GPU) to perform computations on.
    - save_dir (str): Directory to save the trained model.
    - patience (int): Number of epochs to wait for an improvement in validation loss before early stopping.
    """

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training step
        train(model, train_loader, criterion, optimizer, device)

        # Validation step
        val_loss = validate(model, val_loader, criterion, device)
        print(f'Validation Loss after Epoch {epoch + 1}: {val_loss:.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter if improvement
        else:
            patience_counter += 1
            print(f'No improvement in validation loss for {patience_counter} epoch(s).')

        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience_counter} epochs with no improvement.")
            break

        # Step the scheduler to adjust the learning rate
        scheduler.step()

    # Save the MCNN model after training
    save_model(model, save_dir)

