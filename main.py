import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.train_data_utils import prepare_dataloader, initialize_weights, load_model, load_features_used
from scripts.train import train_epochs
from scripts.inference import inference
from utils.config_handler import parse_arguments
from utils.point_cloud_data_utils import read_file_to_numpy, extract_num_classes
import time


def main():
    
    # Parse arguments with defaults from config.yaml
    args = parse_arguments()
    
    # training data file path
    training_data_filepath = args.training_data_filepath  
    
    # training params
    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.patience
    learning_rate = args.learning_rate
    momentum = args.momentum
    step_size = args.learning_rate_decay_epochs
    learning_rate_decay_factor = args.learning_rate_decay_factor
    num_workers = args.num_workers
    save_dir = args.model_save_dir
    '''base_save_dir = args.model_save_dir'''
    
    # inference params
    run_inference_after_training = args.perform_inference_after_training
    use_loaded_model = args.load_model   # whether to load model for inference or train a new one
    loaded_model_path = args.load_model_filepath 
    inference_filepath = args.inference_data_filepath
    
    # feature images creation params
    window_sizes = args.window_sizes
    grid_resolution = 128   # hard-coded value, following reference article 
    
    if use_loaded_model:    # if we run inference on a pre-trained model we don't input features_to_use; instead, we prepare the dataloader on the features we used during the past training
        features_to_use = load_features_used(loaded_model_path)
    else:
        features_to_use = args.features_to_use
    
    # Ensure x, y, z are not included in the selected features
    features_to_use = [feature for feature in features_to_use if feature not in ['x', 'y', 'z']]    
    
    num_classes = extract_num_classes(raw_file_path=training_data_filepath) # determine the number of classes from the raw data    
        
    '''
    many_features = ['intensity', 'return_number', 'number_of_returns', 'ndvi', 'ndwi', 'planarity', 'sphericity', 'linearity', 'theta', 'theta_variance', 'mad', 'delta_z']

    few_features = ['intensity', 'nir', 'delta_z', 'planarity', 'sphericity']
    
    features_for_loop = [many_features, few_features] 
    
    for features_to_use in features_for_loop:
    
        save_dir = base_save_dir + f'_{len(features_to_use)}_features'
        print(f"Training with feature set: {features_to_use}")
    '''   

    num_channels = len(features_to_use)  # Determine the number of channels based on selected features

    # Set device (GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")  
    
    if not use_loaded_model:  # Training
        
        # Prepare DataLoaders for training and validation
        print("Preparing data loaders...")

        train_loader, val_loader = prepare_dataloader(
            batch_size=batch_size,
            data_dir=training_data_filepath,
            window_sizes=window_sizes,
            grid_resolution=grid_resolution,
            features_to_use=features_to_use,
            train_split=0.8,
            num_workers=num_workers,
            shuffle_train=True
        )
        
        data_array, known_features = read_file_to_numpy(data_dir=training_data_filepath, features_to_use=None)   # get the known features from the raw file path.
        
        print(f'Window sizes: {window_sizes}')
        
        print(f'Features read from data file: {known_features}')
        print(f'Selected features to use during training: {features_to_use}')
        
        hyperparameters = {     # store hyperparameters and metadata in dictionary in order to save them together with the model
            'training file': training_data_filepath,
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
        
        # Initialize model 
        print("Initializing MultiScaleCNN (MCNN) model...")
        model = MultiScaleCNN(channels=num_channels, classes=num_classes).to(device)  
        # model.apply(initialize_weights)     # initialize model weights (optional)

        # Set up CrossEntropy loss function
        criterion = nn.CrossEntropyLoss()

        # Set up optimizer (Stochastic Gradient Descent)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                            gamma=learning_rate_decay_factor)

        # Training and validation loop
        print("Starting training process...")
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
        print(f"Training completed in {elapsed_time:.2f} hours")

        # Check if inference is required after training
        if run_inference_after_training:
            print("Starting inference after training...")
            
            print('Preparing inference dataloader...')      # Inference on the test file
            inference_loader, _ = prepare_dataloader(
                    batch_size=batch_size,
                    data_dir='data/training_data/21/test_21.csv',  # test file
                    window_sizes=window_sizes,
                    grid_resolution=grid_resolution,
                    features_to_use=features_to_use,
                    train_split=None,   # prepare the dataloader with the full data for inference (no train/eval split)
                    num_workers=num_workers,
                    shuffle_train=False # we dont want to shuffle data for inference
                )
            
            print('Performing inference...')
            
            '''model_save_folder += f'/_{len(features_to_use)}_features'    # when looping over different features to avoid owerwriting'''
            
            conf_matrix, class_report = inference(
                model=model, 
                dataloader=inference_loader, 
                device=device, 
                class_names=['Grass', 'High Vegetation', 'Building', 'Railway', 'Road', 'Car'], 
                model_save_folder=model_save_folder, 
                save=True
            )
            print(f'Class report output:\n{class_report}')
            print(f'Inference process ended.') 
        
    else:  # Standalone inference (with a loaded model)
        
        print("Starting inference process...")

        # load pre-trained model 
        print(f'Loading pre-trained model from path: {loaded_model_path}')
        model = load_model(model_path=loaded_model_path, device=device, num_channels=num_channels, num_classes=num_classes)
        print('Model loaded successfully')
        
        print('Preparing inference dataloader...')      # Inference on the inference file
        inference_loader, _ = prepare_dataloader(
                batch_size=batch_size,
                data_dir=inference_filepath,  
                window_sizes=window_sizes,
                grid_resolution=grid_resolution,
                features_to_use=features_to_use,
                train_split=None,   # prepare the dataloader with the full data for inference (no train/eval split)
                num_workers=num_workers,
                shuffle_train=False # we dont want to shuffle data for inference
            )
        
        print(f'Performing inference on data contained in {inference_filepath}...')
        conf_matrix, class_report = inference(
            model=model, 
            dataloader=inference_loader, 
            device=device, 
            class_names=['Grass', 'High Vegetation', 'Building', 'Railway', 'Road', 'Car'],  
            model_save_folder=loaded_model_path, 
            # model_save_folder='tests/test_inference/',  # temporarily save in this folder to look at results
            save=True
        )
        
        print(f'Class report output:\n{class_report}')
        print(f'Inference process ended.') 
        
        # send_sms_notification("The model's training has been completed.")


if __name__ == "__main__":
    main()

