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

    def select_device():
        """
        Selects the best available device for PyTorch: CUDA > DirectML > CPU.

        Returns:
        - torch.device: The selected device (CUDA, DirectML, or CPU).
        """
        if torch.cuda.is_available():
            print("Using CUDA device.")
            return torch.device('cuda')
        elif torch.directml.is_available():
            print("CUDA not available. Using DirectML device.")
            return torch.device('dml')
        else:
            print("CUDA and DirectML not available. Using CPU.")
            return torch.device('cpu')

