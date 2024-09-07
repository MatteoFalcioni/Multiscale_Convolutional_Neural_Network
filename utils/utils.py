from datetime import datetime
import os


def get_model_save_path(model_name, base_dir):
    """
    Generates a unique file path for saving the model with its name and a timestamp.

    Args:
    - model_name (str): The name of the model (e.g., 'scnn' or 'mcnn').
    - base_dir (str): The base directory where models will be saved.

    Returns:
    - str: The full path for saving the model.
    """
    # Ensure the save directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Get the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Return the formatted path with model name and timestamp
    return os.path.join(base_dir, f"{model_name}_{timestamp}.pth")
