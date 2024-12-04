'''NOTA BENE: attento quando carichi i center points dal dataloader, per generare le griglie mettili in float64. 
Sennò le griglie non matchano tra cpu e gpu'''

from tqdm import tqdm
from scripts.vectorized_grid_gen import vectorized_generate_multiscale_grids


def train(model, dataloader, criterion, optimizer, device, shared_objects):
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
        raw_points, labels = raw_points.to(device), labels.to(device)

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

        small_grids, medium_grids, large_grids = grids[:, 0], grids[:, 1], grids[:, 2]

        optimizer.zero_grad()
        outputs = model(small_grids, medium_grids, large_grids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Batch Loss': loss.item(), 'Avg Loss': total_loss / (i + 1)})

    return total_loss / len(dataloader)
