import argparse
import yaml
import ast

# Custom function to parse the window_sizes argument
def parse_window_sizes(value):
    """
    Parses a string that represents a list (e.g., '[10, 20, 30]')
    and converts it into the desired tuple format:
    Example: '[10, 20, 30]' -> [('small', 10.0), ('medium', 20.0), ('large', 30.0)]
    """
    try:
        # Parse the string input to a list
        sizes = ast.literal_eval(value)
        if not isinstance(sizes, list) or len(sizes) != 3:
            raise ValueError("Input must be a list of exactly three numeric values.")
        return [('small', float(sizes[0])), ('medium', float(sizes[1])), ('large', float(sizes[2]))]
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid format. Use a list like '[10, 20, 30]'.")



def load_config(file_path='code/config.yaml'):
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
    
    parser.add_argument('--dataset_filepath', type=str, default=config.get('dataset_filepath', 'data/datasets/sampled_full_dataset/sampled_data_5251681.csv'),
                        help='File path to the full dataset file.')
    
    parser.add_argument('--training_data_filepath', type=str, default=config.get('training_data_filepath', 'data/datasets/train_dataset.csv'),
                        help='File path to thed ata to be used during training.')
    
    parser.add_argument('--window_sizes',
                        type=parse_window_sizes,
                        metavar='[SMALL,MEDIUM,LARGE]',
                        default=[('small', 2.5), ('medium', 5.0), ('large', 10.0)],
                        help="List of three window sizes for grid generation, e.g., '[10, 20, 30]'.")

    parser.add_argument('--features_to_use', type=str, nargs='+', default=config.get('features_to_use'),
                    help='List of feature names to use for training (e.g., intensity red green blue)')
    
    parser.add_argument('--model_save_dir', type=str, default=config.get('model_save_dir', 'models/saved/'),
                        help='Directory to save trained models')
    
    parser.add_argument('--save_model', action='store_true', default=config.get('save_model', True),
                        help='If set, save the trained model.')
    
    parser.add_argument('--evaluate_model_after_training', action='store_true', default=config.get('evaluate_model_after_training', False),
                        help='If set, evaluates the model directly after training it.')
    
    parser.add_argument('--perform_evaluation', action='store_true', default=config.get('perform_evaluation', False),
                    help='If set, evaluates the loaded model.')

    parser.add_argument('--load_model_filepath', type=str,
                        default=config.get('load_model_filepath', 'models/saved/mcnn_model_20241015_005511/model.pth'),
                        help='File path to the pre-trained model to be loaded.')
    
    parser.add_argument('--evaluation_data_filepath', type=str, default=config.get('evaluation_data_filepath', 'data/datasets/eval_dataset.csv'),
                        help='File path to the data to be used for evaluating the model.')
    
    parser.add_argument('--predict_labels', action='store_true', default=config.get('predict', False),
                        help='If set, runs predictions on a given file.')
    
    parser.add_argument('--file_to_predict', type=str,
                        default=config.get('file_to_predict', 'data/chosen_tiles/'),
                        help='File path to the file we need to run predictions on.')
    
    # Parsing arguments
    args = parser.parse_args()
    
    # If dataset_filepath was not explicitly set on the command line
    if not args.dataset_filepath:  
        # 1) Check if training_data_filepath was provided
        if args.training_data_filepath:
            args.dataset_filepath = args.training_data_filepath
        
        # 2) Otherwise, if evaluation_data_filepath was provided
        elif args.evaluation_data_filepath:
            args.dataset_filepath = args.evaluation_data_filepath
            

    return args


