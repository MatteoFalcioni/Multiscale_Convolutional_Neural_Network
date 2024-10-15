import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.train_data_utils import prepare_dataloader, initialize_weights, load_model 
from scripts.train import train_epochs
from scripts.inference import inference
from utils.config_handler import parse_arguments
from utils.point_cloud_data_utils import read_file_to_numpy, extract_num_classes, get_feature_indices
import time


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
    num_workers = args.num_workers
    
    # inference params
    use_loaded_model = args.load_model   # whether to load model for inference or train a new one
    model_path = args.load_model_filepath
    
    data_array, known_features = read_file_to_numpy(data_dir=data_dir, features_to_use=None)   # get the known features from the raw file path.

    num_channels = len(features_to_use)  # Determine the number of channels based on selected features
    num_classes = extract_num_classes(raw_file_path=data_dir) # determine the number of classes from the raw data
    
    print(f'window sizes: {window_sizes}')
    
    print(f'features contained in raw data file: {known_features}')
    print(f'selected features to use during training: {features_to_use}')
    
    hyperparameters = {     # store hyperparameters in dictionary in order to save them together with the model
        'number of total points' : data_array.shape[0],
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
        num_workers=num_workers
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
    start_time = time.time()

    model_save_folder = train_epochs(
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

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time:.2f} seconds")

    # Run inference 
    print("Starting inference process...")

    # load pre-trained model if chosen by the user
    if use_loaded_model:
        model = load_model(model_path=model_path, device=device, num_channels=num_channels, num_classes=num_classes)

    conf_matrix, class_report = inference(model=model, dataloader=val_loader, device=device, class_names=['Grass', 'High Vegetation', 'Building', 'Railway', 'Road', 'Car'], model_save_folder=model_save_folder, save=True)
    
    print(f'Inference process ended.') 


if __name__ == "__main__":
    main()

