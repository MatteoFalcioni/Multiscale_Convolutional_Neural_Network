import argparse
import yaml
import ast


def load_config(file_path='config.yaml'):
    """
    Load configuration from a YAML file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_arguments():
    """
    Sets up the argument parser for training configurations and parses command-line arguments.

    Returns:
    - argparse.Namespace: Parsed arguments as a namespace object.
    """

    # Load configuration defaults from the config file
    config = load_config()

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train a deep learning model.")

    # Adding arguments for hyperparameters and configurations with defaults from the config file
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 16), help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=config.get('epochs', 20),
                        help='Number of epochs to wait for an improvement in validation loss before early stopping')
    
    parser.add_argument('--patience', type=int, default=config.get('patience', 3),
                        help='Number of training epochs')
    
    parser.add_argument('--learning_rate', type=float, default=config.get('learning_rate', 0.01),
                        help='Learning rate for the optimizer')
    
    parser.add_argument('--learning_rate_decay_factor', type=float,
                    default=config.get('learning_rate_decay_factor', 0.0005),
                    help='Learning rate decay factor')
    
    parser.add_argument('--learning_rate_decay_epochs', type=int, default=config.get('learning_rate_decay_epochs', 1),
                        help='Epochs interval to decay learning rate')
    
    parser.add_argument('--momentum', type=float, default=config.get('momentum', 0.9),
                        help='Momentum factor for the optimizer')
    
    parser.add_argument('--num_workers', type=int, default=config.get('num_workers', 16),
                        help='Number of workers for parallel processing')
    
    parser.add_argument('--training_data_filepath', type=str, default=config.get('training_data_filepath', 'data/training_data/21/test_21.csv'),
                        help='File path to thed ata to be used during training.')
    
    parser.add_argument('--window_sizes', type=ast.literal_eval,
                    default=config.get('window_sizes', [('small', 2.5), ('medium', 5.0), ('large', 10.0)]),
                    help="List of window sizes for grid generation (e.g., [('small', 2.5), ('medium', 5.0), ('large', 10.0)])")
    
    parser.add_argument('--features_to_use', type=str, nargs='+', default=config.get('features_to_use'),
                    help='List of feature names to use for training (e.g., intensity red green blue)')
    
    parser.add_argument('--model_save_dir', type=str, default=config.get('model_save_dir', 'models/saved/'),
                        help='Directory to save trained models')
    
    parser.add_argument('--save_model', action='store_true', default=config.get('save_model', True),
                        help='If set, save the trained model.')
    
    parser.add_argument('--perform_inference_after_training', action='store_true', default=config.get('perform_inference_after_training', False),
                        help='If set, performs inference directly after training the model.')
    
    parser.add_argument('--load_model', action='store_true', default=config.get('load_model', False),
                        help='If set, loads model from load_model_filepath to perform inference')
    
    parser.add_argument('--load_model_filepath', type=str,
                        default=config.get('load_model_filepath', 'models/saved/mcnn_model_20241015_005511/model.pth'),
                        help='File path to the pre-trained model to be loaded.')
    
    parser.add_argument('--inference_data_filepath', type=str, default=config.get('inference_data_filepath', 'data/raw'),
                        help='File path to the data to be used for inference.')
    
    
    
    

    # Parsing arguments
    args = parser.parse_args()

    return args


