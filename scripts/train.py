import torch
from utils.plot_utils import plot_loss
from utils.train_data_utils import save_model
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, device):
    """
    Trains the MultiScale CNN model for one epoch.

    Args:
    - model (nn.Module): The PyTorch model to be trained.
    - dataloader (DataLoader): The DataLoader object containing the training data.
    - criterion (nn.Module): The loss function to compute the error between predicted and true labels.
    - optimizer (torch.optim.Optimizer): The optimizer used to update the model's weights (e.g., SGD, Adam).
    - device (torch.device): The device (CPU or GPU) where the computations will be performed.

    Returns:
    - float: The average loss for the entire epoch, computed as the total weighted loss divided by the total number of processed samples.

    Raises:
    - ValueError: If NaN or Inf loss is encountered during training.
    """
    model.train()  # Set model to training mode
    total_loss = 0.0  # This will accumulate the loss for the entire epoch
    total_samples = 0  # This will keep track of the total number of samples processed

    # Initialize tqdm progress bar for the training loop
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', leave=False)

    for i, batch in progress_bar:
        if batch is None:  # Skip if the batch is None
            continue
        
        # Unpack the batch
        small_grids, medium_grids, large_grids, labels = batch

        
        small_grids, medium_grids, large_grids, labels = (
            small_grids.to(device), medium_grids.to(device), large_grids.to(device), labels.to(device)
        )
        
        batch_size = labels.size(0) # get the actual batch size

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(small_grids, medium_grids, large_grids)

        # Compute loss
        loss = criterion(outputs, labels)

        # Check for NaN or Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("NaN or Inf loss encountered during training. Stopping training.")

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size  # Accumulate weighted loss (by batch size)
        total_samples += batch_size  # Accumulate the number of samples processed

        # Update progress bar every 10 batches
        if (i + 1) % 10 == 0:
            progress_bar.set_postfix({
                'Current Batch Loss': loss.item(),  # loss.item() is the loss for the current batch
                'Avg Loss So Far': total_loss / total_samples   # total_loss / total_samples is the the avg loss per point proces so far in the epoch
            })

    # Return the average loss over all samples
    return total_loss / total_samples


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set and returns the average loss.

    Args:
    - model (nn.Module): The PyTorch model to be trained.
    - dataloader (DataLoader): DataLoader object containing the validation data.
    - criterion (nn.Module): The loss function to optimize.
    - device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    - float: the average validation loss for each epoch
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0  # To accumulate the total loss
    total_samples = 0  # To track the total number of samples processed

    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            if batch is None:  # Skip if the batch is None
                continue
            
            # Unpack the batch
            small_grids, medium_grids, large_grids, labels = batch

            small_grids, medium_grids, large_grids, labels = (
                small_grids.to(device), medium_grids.to(device), large_grids.to(device), labels.to(device)
            )

            # Get the batch size
            batch_size = labels.size(0)

            # Forward pass
            outputs = model(small_grids, medium_grids, large_grids)
            loss = criterion(outputs, labels)

            # Accumulate the loss weighted by batch size
            val_loss += loss.item() * batch_size
            total_samples += batch_size

    # Return the average validation loss per sample
    return val_loss / total_samples


def train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device, model_save_dir='models/saved/', plot_dir='results/plots/', save=False, used_features=None, hyperparameters=None):
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
    - plot_dir (str): Directory to save the loss plots. Default is 'results/plots/'.
    - save (bool): boolean value to allow or disallow saving of the model after training. Default is False.
    - used_features (list): selected features used during training, to be saved together with the model- Default is None.
    - hyperparameters (dict): dictionary of hyperparameters (name, values) used during training, to be saved together with the model- Default is None.
    
    Returns:
    - model_save_foder (str): Name of the folder where the model has been saved. Needed later for inference.
    """

    train_losses = []  # To store training losses
    val_losses = []  # To store validation losses

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training step
        train_loss = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validation step
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

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
    if save:
        print("saving trained model...")
        model_save_folder = save_model(model, model_save_dir, used_features=used_features, hyperparameters=hyperparameters)
        print("model saved successfully.")

    # Plot the losses at the end of training
    plot_loss(train_losses, val_losses, save_dir=model_save_folder)

    return model_save_folder



