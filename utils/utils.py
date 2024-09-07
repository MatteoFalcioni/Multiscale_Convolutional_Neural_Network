from datetime import datetime
import os
import torch


def save_model(model, save_dir):
    """
    Saves the MCNN model in the specified directory with a timestamp.

    Args:
    - model (nn.Module): The MCNN model to be saved.
    - save_dir (str): Directory where the model will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a filename with the current date and time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(save_dir, f"mcnn_{timestamp}.pth")

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
