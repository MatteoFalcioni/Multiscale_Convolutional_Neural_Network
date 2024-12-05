import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.train_data_utils import prepare_dataloader
from scripts.train import train_epochs
from utils.point_cloud_data_utils import extract_num_classes, read_file_to_numpy
import time


def train_model(full_data_filepath, features_to_use, batch_size, epochs, patience, learning_rate, momentum, step_size, learning_rate_decay_factor, num_workers, save_dir, device, window_sizes, grid_resolution=128, training_data_filepath=None):
    """
    Trains a MultiScaleCNN (MCNN) model for point cloud classification using the provided training data and hyperparameters.

    This function sets up the model, prepares data loaders, and runs the training and validation loops. 
    It also manages the hyperparameters and saves the trained model.

    Args:
        - full_data_filepath (str): Path to the full dataset.
        - features_to_use (str): Features to use for feature images generation.
        - batch_size (int): Batch size to use for training the model.
        - epochs (int): Number of epochs to train the model.
        - patience (int): The number of epochs with no improvement before training is stopped.
        - learning_rate (float): Initial learning rate for the optimizer.
        - momentum (float): Momentum factor for the SGD optimizer.
        - step_size (int): Number of epochs between each learning rate decay step.
        - learning_rate_decay_factor (float): Factor by which the learning rate is multiplied during each decay step.
        - num_workers (int): Number of CPU workers to use for loading data in parallel.
        - save_dir (str): Directory where the trained model and hyperparameters will be saved.
        - device (torch.device): Device on which to train the model (CPU or GPU).
        - window_sizes (list): List of tuples for grid window sizes (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)]).
        - grid_resolution (int): Resolution of the grid used for preparing input data for the model.
        - training_data_filepath (str, optional): Path to the CSV training file containing coordinates of points to be selected from the full data. If None, all are selected.


    Returns:
        - model (MultiScaleCNN) : The trained model
        - model_save_folder (str): Directory where the trained model has been saved.
    """
    # Ensure (additional check) that x, y, z are not included in the selected features
    features_to_use = [feature for feature in features_to_use if feature not in ['x', 'y', 'z']]    
    
    num_classes = extract_num_classes(raw_file_path=full_data_filepath)   # determine the number of classes from the full dataset   
    num_channels = len(features_to_use)  # Determine the number of channels based on selected features  

    # Prepare DataLoaders for training and validation
    train_loader, val_loader = prepare_dataloader(
        batch_size=batch_size,
        data_filepath=full_data_filepath,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        features_to_use=features_to_use,
        train_split=0.8,
        num_workers=num_workers,
        shuffle_train=True,
        subset_file=training_data_filepath
    )
    
    full_data_array, full_known_features = read_file_to_numpy(data_dir=full_data_filepath, features_to_use=None)   # get the known features from the full dataset
    total_num_points = full_data_array.shape[0]

    print(f'Loaded full point cloud data with {total_num_points} points')

    print(f'Features read from full data file: {full_known_features}\n')
    print(f'Selected features to use during training: {features_to_use}\n')
    print(f'Window sizes: {window_sizes}\n')

    print(f'Number of unique classes read from full data file: {num_classes}\n')


    if training_data_filepath is not None: # check subset file 
        subset_array, subset_features = read_file_to_numpy(data_dir=training_data_filepath, features_to_use=None)
        num_subset = subset_array.shape[0]
        num_subset_classes = extract_num_classes(raw_file_path=training_data_filepath)
        print(f"\nSince a subset of the full dataset was selected, feature images will be generated only for points contained in {training_data_filepath}, corresponding to {num_subset} / {total_num_points} points.")
        assert full_known_features == subset_features, f"Full training data features do not match the features from the subset."
        assert num_subset_classes == num_classes, f"Number of unique classes (labels) doesn't match between full data file and subset file."
    else: 
        training_data_filepath = "No subset selected"  # write this in hyperparams


    hyperparameters = {     # store hyperparameters and metadata in dictionary in order to save them together with the model
        'dataset file': full_data_filepath,
        'training file' : training_data_filepath,
        'num_classes' : num_classes,
        'number of total points' : total_num_points,
        'window_sizes' : window_sizes,
        'grid_resolution': grid_resolution,
        'batch_size': batch_size,
        'epochs' : epochs,
        'patience' : patience,
        'learning_rate' : learning_rate,
        'momentum' : momentum,
        'step_size' : step_size,
        'learning_rate_decay_factor' : learning_rate_decay_factor,
        'num_workers' : num_workers
    }
    
    # Initialize model 
    model = MultiScaleCNN(channels=num_channels, classes=num_classes).to(device)

    # Set up CrossEntropy loss function
    criterion = nn.CrossEntropyLoss()

    # Set up optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                        gamma=learning_rate_decay_factor)

    # Training and validation loop
    print("\n==============================Starting training==============================\n", flush=True)
    start_time = time.time()

    model_save_folder = train_epochs(
                                        model=model,
                                        train_loader=train_loader,
                                        val_loader=val_loader,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        epochs=epochs,
                                        patience=patience,
                                        device=device,
                                        save=True,
                                        model_save_dir=save_dir,
                                        used_features=features_to_use,
                                        hyperparameters=hyperparameters
                                    )

    end_time = time.time()
    elapsed_time = (end_time - start_time) /3600   # in hours
    print(f"\n========================Training completed in {elapsed_time:.2f} hours========================\n", flush=True)

    return model, model_save_folder
