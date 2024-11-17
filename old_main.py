import torch
import torch.nn as nn
import torch.optim as optim
from models.mcnn import MultiScaleCNN
from utils.train_data_utils import prepare_dataloader, load_model, load_features_used
from scripts.train import train_epochs
from scripts.inference import evaluate_model, predict
from utils.config_handler import parse_arguments
from utils.point_cloud_data_utils import read_file_to_numpy, extract_num_classes
import time
import glob
import os


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
    
    # inference params
    run_inference_after_training = args.perform_inference_after_training
    use_loaded_model = args.load_model   # whether to load model for inference or train a new one
    loaded_model_path = args.load_model_filepath 
    evaluation_data_filepath = args.evaluation_data_filepath
    predict = args.predict
    file_to_predict = args.file_to_predict
    
    # feature images creation params
    window_sizes = args.window_sizes
    grid_resolution = 128   # hard-coded value, following reference article 

    # Set device (GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if evaluate_model or predict:    # if we run inference on a pre-trained model we don't input features_to_use; instead, we prepare the dataloader on the features we used during the past training
        features_to_use = load_features_used(loaded_model_path)
    else:
        features_to_use = args.features_to_use
    
    # Ensure (additional check) that x, y, z are not included in the selected features
    features_to_use = [feature for feature in features_to_use if feature not in ['x', 'y', 'z']]    
    
    num_classes = extract_num_classes(raw_file_path=training_data_filepath)     # determine the number of classes from the data    

    num_channels = len(features_to_use)  # Determine the number of channels based on selected features  
    
    if not use_loaded_model:  # Training
        
        # Prepare DataLoaders for training and validation
        print("Preparing data loaders...\n")

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
        
        print(f'Window sizes: {window_sizes}\n')
        
        print(f'Features read from data file: {known_features}\n')
        print(f'Selected features to use during training: {features_to_use}\n')
        
        hyperparameters = {     # store hyperparameters and metadata in dictionary in order to save them together with the model
            'training file': training_data_filepath,
            'num_classes' : num_classes,
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
        print("Initializing MultiScaleCNN (MCNN) model...\n")
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
        print("--------------------------------Starting training process----------------------------------")
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
        print(f"-------------------------Training completed in {elapsed_time:.2f} hours-------------------------\n")

        # Check if inference is required after training
        if run_inference_after_training:
            
            # Inference on the test file
            inference_loader, _ = prepare_dataloader(
                    batch_size=batch_size,
                    data_dir='data/training_data/21/test_21.csv',  
                    window_sizes=window_sizes,
                    grid_resolution=grid_resolution,
                    features_to_use=features_to_use,
                    train_split=None,   # prepare the dataloader with the full data for inference (no train/eval split)
                    num_workers=num_workers,
                    shuffle_train=False  # we dont want to shuffle data for inference
                )
            
            conf_matrix, class_report = evaluate_model(
                model=model, 
                dataloader=inference_loader, 
                device=device, 
                class_names=['Grass', 'High Vegetation', 'Building', 'Railway', 'Road', 'Car'], 
                model_save_folder=model_save_folder, 
                inference_file_path=inference_filepath, 
                save=True
            )
            print(f'Class report output:\n{class_report}')
            print(f'\nInference process ended.') 
        
    else:  
        
        print("--------------------------------Starting inference process...--------------------------------")

        # load pre-trained model 
        model = load_model(model_path=loaded_model_path, device=device, num_channels=num_channels)
        
        # Run predictions
        predict(file_path=file_to_predict, model=model, model_path=loaded_model_path, device=device,
                batch_size=batch_size, window_sizes=window_sizes, grid_resolution=grid_resolution, features_to_use=features_to_use,
                num_workers=num_workers, tile_size=125)
        

if __name__ == "__main__":
    main()




'''#print('Preparing inference dataloader...')      
        inference_loader, _ = prepare_dataloader(
                batch_size=batch_size,
                data_dir=inference_filepath,  
                window_sizes=window_sizes,
                grid_resolution=grid_resolution,
                features_to_use=features_to_use,
                train_split=None,   # prepare the dataloader with the full data for inference (no train/eval split)
                num_workers=num_workers,
                shuffle_train=False   # we dont want to shuffle data for inference
            )
        
        print(f'Performing inference on data contained in {inference_filepath}...')
        
        conf_matrix, class_report = inference(
            model=model, 
            dataloader=inference_loader, 
            device=device, 
            class_names=['Grass', 'High Vegetation', 'Building', 'Railway', 'Road', 'Car'],  
            model_save_folder=loaded_model_path, 
            inference_file_path=inference_filepath,
            save=True
        )
        print(f'Class report output:\n{class_report}')
        print(f'Inference process ended.') 
        
        # Directory containing LAS files
        directory = 'data/chosen_tiles/32_687000_4930000_FP21_125'  
        las_files = glob.glob(os.path.join(directory, '*.las'))

        # Total number of files
        total_files = len(las_files) 

        # Loop over the LAS files using glob
        for index, file_path in enumerate(las_files, start=1):
            print(f"************ Processing file {index}/{total_files}: {file_path} ************\n")
            
            print('Preparing inference dataloader...\n')      
            inference_loader, _ = prepare_dataloader(
                    batch_size=batch_size,
                    data_dir=file_path,  
                    window_sizes=window_sizes,
                    grid_resolution=grid_resolution,
                    features_to_use=features_to_use,
                    train_split=None,   # prepare the dataloader with the full data for inference (no train/eval split)
                    num_workers=num_workers,
                    shuffle_train=False # we dont want to shuffle data for inference
                )
            
            print(f'Performing inference on data contained in {file_path}...\n')
        
            file_with_predictions = inference_without_ground_truth(
                        model=model, 
                        dataloader=inference_loader, 
                        device=device, 
                        data_file=file_path, 
                        model_path=loaded_model_path
                    )

            print(f'Inference process completed successfully for file {file_path}.\nLas file with predicted labels saved at {file_with_predictions}\n')
        '''

