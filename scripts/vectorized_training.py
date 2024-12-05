'''NOTA BENE: attento quando carichi i center points dal dataloader, per generare le griglie mettili in float64. 
Sennò le griglie non matchano tra cpu e gpu'''

from tqdm import tqdm
from scripts.vectorized_gpu_grid_gen import vectorized_generate_multiscale_grids
import torch
from utils.plot_utils import plot_loss
from utils.train_data_utils import save_model


def vec_train(model, dataloader, criterion, optimizer, device, shared_objects):
    """
    Trains the model for one epoch.

    Args:
    - model (nn.Module): The PyTorch model.
    - dataloader (DataLoader): DataLoader for the training data.
    - criterion (nn.Module): Loss function.
    - optimizer (torch.optim.Optimizer): Optimizer for model parameters.
    - device (torch.device): Device for computations.
    - shared_objects (dict): Shared utilities like KDTree, tensor_full_data, etc.

    Returns:
    - float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Training', leave=False)

    for i, (center_points, labels, _) in progress_bar:  
        center_points, labels = center_points.to(device), labels.to(device)

        # Generate grids on-the-fly
        grids = vectorized_generate_multiscale_grids(
            center_points=center_points,
            window_sizes=shared_objects['window_sizes_tensor'],
            grid_resolution=shared_objects['grid_resolution'],
            gpu_tree=shared_objects['gpu_tree'],
            tensor_data_array=shared_objects['tensor_full_data'],
            feature_indices_tensor=shared_objects['feature_indices_tensor'],
            device=device
        )

        # Split grids and cast to float32
        small_grids, medium_grids, large_grids = (
            grids[:, 0].to(torch.float32),
            grids[:, 1].to(torch.float32),
            grids[:, 2].to(torch.float32)
        )

        optimizer.zero_grad()
        outputs = model(small_grids, medium_grids, large_grids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Batch Loss': loss.item(), 'Avg Loss': total_loss / (i + 1)})

    return total_loss / len(dataloader)


def vec_validate(model, dataloader, criterion, device, shared_objects):
    """
    Validates the model on the validation dataset.

    Args:
    - model (nn.Module): The PyTorch model.
    - dataloader (DataLoader): DataLoader for the validation data.
    - criterion (nn.Module): Loss function.
    - device (torch.device): Device for computations.
    - shared_objects (dict): Shared utilities.

    Returns:
    - float: Average validation loss.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for center_points, labels, _ in tqdm(dataloader, desc="Validating", leave=False):
            center_points, labels = center_points.to(device), labels.to(device)

            grids = vectorized_generate_multiscale_grids(
                center_points=center_points,
                window_sizes=shared_objects['window_sizes_tensor'],
                grid_resolution=shared_objects['grid_resolution'],
                gpu_tree=shared_objects['gpu_tree'],
                tensor_data_array=shared_objects['tensor_full_data'],
                feature_indices_tensor=shared_objects['feature_indices_tensor'],
                device=device
            )

            # Split grids and cast to float32
            small_grids, medium_grids, large_grids = (
                grids[:, 0].to(torch.float32),
                grids[:, 1].to(torch.float32),
                grids[:, 2].to(torch.float32)
            )
            outputs = model(small_grids, medium_grids, large_grids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def vec_train_epochs(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, device, shared_objects, model_save_dir='models/saved/', save=False, used_features=None, hyperparameters=None):
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
    - shared_objects (dict): Dictionary containing reusable utilities (KDTree, full data tensor, feature indices, etc.).
    - save_dir (str): Directory to save the trained model.
    - patience (int): Number of epochs to wait for an improvement in validation loss before early stopping.
    - save (bool): boolean value to allow or disallow saving of the model after training. Default is False.
    - used_features (list): selected features used during training, to be saved together with the model- Default is None.
    - hyperparameters (dict): dictionary of hyperparameters (name, values) used during training, to be saved together with the model- Default is None.

    Returns:
    - model_save_folder (str): Name of the folder where the model has been saved. Needed later for inference.
    """
    train_losses = []  # To store training losses
    val_losses = []  # To store validation losses

    best_val_loss = float('inf')
    patience_counter = 0

    # Extract shared objects
    """gpu_tree = shared_objects['gpu_tree']
    tensor_full_data = shared_objects['tensor_full_data']
    feature_indices_tensor = shared_objects['feature_indices_tensor']
    window_sizes_tensor = shared_objects['window_sizes_tensor']
    grid_resolution = shared_objects['grid_resolution']"""

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')

        # Training step
        train_loss = vec_train(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            shared_objects
        )
        train_losses.append(train_loss)

        # Validation step
        val_loss = vec_validate(
            model,
            val_loader,
            criterion,
            device,
            shared_objects
        )
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter if improvement
            print(f'Validation loss is decreasing. Current value: {val_loss:.6f}. Continuing training... ')
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
        print("Saving trained model...")
        model_save_folder = save_model(model, model_save_dir, used_features=used_features, hyperparameters=hyperparameters)
        print("Model saved successfully.")

    # Plot the losses at the end of training
    plot_loss(train_losses, val_losses, save_dir=model_save_folder)

    return model_save_folder
