import torch
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model
from scripts.inference import predict
from utils.config_handler import parse_arguments
from utils.train_data_utils import load_parameters, load_model


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
    evaluate_model_after_training = args.evaluate_model_after_training
    loaded_model_path = args.load_model_filepath 
    perform_evaluation = args.perform_evaluation
    evaluation_data_filepath = args.evaluation_data_filepath
    predict_labels = args.predict_labels
    file_to_predict = args.file_to_predict
    
    # feature images creation params
    features_to_use = args.features_to_use  # features to use during training
    window_sizes = args.window_sizes
    grid_resolution = 128   # hard-coded value, following reference article
    
    # Set device (GPU if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if predict_labels and perform_evaluation:
        raise ValueError("You can either predict new labels or evaluate the model's performance. Please set only one among predict_labels and perform_evaluation as True.")

    if not predict_labels and not perform_evaluation:

        # training
        model, model_save_folder = train_model(training_data_filepath=training_data_filepath,
                                                                features_to_use=features_to_use,
                                                                batch_size = batch_size,
                                                                epochs = epochs,
                                                                patience = patience,
                                                                learning_rate = learning_rate,
                                                                momentum = momentum,
                                                                step_size = step_size,
                                                                learning_rate_decay_factor = learning_rate_decay_factor,
                                                                num_workers = num_workers,
                                                                save_dir = save_dir,
                                                                device = device,
                                                                window_sizes=window_sizes,
                                                                grid_resolution=grid_resolution,
                                                                subset_file=)
        
        if evaluate_model_after_training:
            # perform evaluation after training 
            evaluate_model(batch_size=batch_size, 
                           data_dir=evaluation_data_filepath, 
                           window_sizes=window_sizes, 
                           grid_resolution=grid_resolution, 
                           features_to_use=features_to_use, 
                           num_workers=num_workers, 
                           model=model, 
                           device=device, 
                           model_save_folder=model_save_folder, 
                           evaluation_data_filepath=evaluation_data_filepath)

    elif perform_evaluation:

        # load features used during model training and the respetctive number of channels and window sizes
        loaded_features, num_loaded_channels, window_sizes = load_parameters(loaded_model_path)
        # load the pre-trained model
        loaded_model = load_model(model_path=loaded_model_path, device=device, num_channels=num_loaded_channels)

        # Evaluate loaded model
        evaluate_model(batch_size=batch_size, 
                        data_dir=evaluation_data_filepath, 
                        window_sizes=window_sizes, 
                        grid_resolution=grid_resolution, 
                        features_to_use=loaded_features, 
                        num_workers=num_workers, 
                        model=loaded_model, 
                        device=device, 
                        model_save_folder=model_save_folder, 
                        evaluation_data_filepath=evaluation_data_filepath)
        
    elif predict_labels:

        # load features used during model training and the respetctive number of channels and window sizes
        loaded_features, num_loaded_channels, window_sizes = load_parameters(loaded_model_path)
        # load pre-trained model 
        loaded_model = load_model(model_path=loaded_model_path, device=device, num_channels=num_loaded_channels)
        
        # Run predictions
        predict(file_path=file_to_predict, model=loaded_model, model_path=loaded_model_path, device=device,
                batch_size=batch_size, window_sizes=window_sizes, grid_resolution=grid_resolution, features_to_use=loaded_features,
                num_workers=num_workers, tile_size=125)
        
if __name__ == "__main__":
    main()



