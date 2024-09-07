import argparse


def parse_arguments():
    """
    Sets up the argument parser for training configurations and parses command-line arguments.

    Returns:
    - argparse.Namespace: Parsed arguments as a namespace object.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train a deep learning model.")

    # Adding arguments for hyperparameters and configurations
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--learning_rate_decay_epochs', type=int, default=5,
                        help='Epochs interval to decay learning rate')
    parser.add_argument('--learning_rate_decay_gamma', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--save_dir', type=str, default='models/saved/', help='Directory to save trained models')

    # Parsing arguments
    args = parser.parse_args()

    return args
