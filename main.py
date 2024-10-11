import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.train_data_utils import prepare_dataloader, initialize_weights
from scripts.train import train_epochs
from scripts.inference import inference
from utils.config_handler import parse_arguments
from utils.point_cloud_data_utils import read_file_to_numpy, extract_num_classes, get_feature_indices
import numpy as np


def main():
    # Parse arguments with defaults from config.yaml
    args = parse_arguments()
    
    # raw data file path
    data_dir = args.raw_data_filepath   
    
    # feature images creation params
    window_sizes = args.window_sizes
    features_to_use = args.features_to_use
    grid_resolution = 128   # hard-coded value, following reference article
    
    # training params
    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.patience
    learning_rate = args.learning_rate
    momentum = args.momentum
    step_size = args.learning_rate_decay_epochs
    learning_rate_decay_factor = args.learning_rate_decay_factor
    
    # inference params
    load_model = args.load_model   # whether to load model for inference or train a new one
    model_path = args.load_model_filepath
    
    data_array, known_features = read_file_to_numpy(data_dir=data_dir, features_to_use=None)   # get the known features from the raw file path.

    num_channels = len(features_to_use)  # Determine the number of channels based on selected features
    num_classes = extract_num_classes(raw_file_path=data_dir) # determine the number of classes from the raw data
    
    feature_indices = get_feature_indices(features_to_use=features_to_use, known_features=known_features)   # get the corresponding indices of the chosen features in known features
    
    print(f'window sizes: {window_sizes}')
    
    print(f'features contained in raw data file: {known_features}')
    print(f'selected features to use during training: {features_to_use}')
    
    hyperparameters = {     # store them in dictionary in order to save them together with the model
        'number of points' : data_array.shape[0],
        'window_sizes' : window_sizes,
        'grid_resolution': grid_resolution,
        'batch_size': batch_size,
        'epochs' : epochs,
        'patience' : patience,
        'learning_rate' : learning_rate,
        'momentum' : momentum,
        'step_size' : step_size,
        'learning_rate_decay_factor' : learning_rate_decay_factor
    }

    # Set device (GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  

    # Initialize model 
    print("Initializing MultiScaleCNN (MCNN) model...")
    model = MultiScaleCNN(channels=num_channels, classes=num_classes).to(device)  
    model.apply(initialize_weights)     # initialize model weights (optional, but recommended)

    # Prepare DataLoader
    print("Preparing data loaders...")

    train_loader, val_loader = prepare_dataloader(
        batch_size=batch_size,
        data_dir=data_dir,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        features_to_use=features_to_use,
        train_split=0.8,
        num_workers=16
    )

    # Set up CrossEntropy loss function
    criterion = nn.CrossEntropyLoss()

    # Set up optimizer (Stochastic Gradient Descent)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                          gamma=learning_rate_decay_factor)

    # Training loop
    print("Starting training process...")
    train_epochs(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs,
        patience,
        device,
        save=True,
        plot_dir='results/plots/',
        model_save_dir="models/saved/",
        used_features=features_to_use,
        hyperparameters=hyperparameters
    )

    print("Training finished")

    # Run inference on a sample
    print("Starting inference process...")
    
    if load_model:
        loaded_model = MultiScaleCNN(channels=num_channels, classes=num_classes).to(device)
        # Load the saved model state dictionary
        loaded_model.load_state_dict(torch.load(model_path, map_location=device))
        model = loaded_model
        # Set model to evaluation mode (important for inference)
        model.eval()

    # use a file that's never been seen by the model for inference
    inference_file = 'data/sampled/sampled_data_inference_1000.csv'
    data_array_inference, _ = read_file_to_numpy(data_dir=inference_file, features_to_use=features_to_use)  # feature indices and num classes in inference should match features in train/eval data

    true_labels, predicted_labels = inference(
        model=model,
        data_array=data_array_inference,
        window_sizes=window_sizes,
        grid_resolution=grid_resolution,
        feature_indices=feature_indices,
        save_file='results/inference/inference.csv',
        subsample_size=None,
        device=device
    )
    
    print(f'Inference process ended.')  # Ground truth labels: {true_labels} \n Predicted labels: {predicted_labels}


if __name__ == "__main__":
    main()



