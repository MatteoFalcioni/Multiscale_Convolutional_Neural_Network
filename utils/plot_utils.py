import matplotlib.pyplot as plt
import os
from datetime import datetime


def plot_loss(train_losses, val_losses, save_dir='results/plots/'):
    """
    Plots the training and validation loss over epochs and saves the plot with a label for the model.

    Args:
    - train_losses (list of float): List containing the training loss values for each epoch.
    - val_losses (list of float): List containing the validation loss values for each epoch.
    - save_dir (str): Directory where to save the plot. Defaults to 'results/plots/'.
    """
    # Create the 'plots' folder inside the results directory
    os.makedirs(save_dir, exist_ok=True)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define the path to save the plot
    plot_path = os.path.join(save_dir, f"MCNN_loss_plot_{timestamp}.png")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linestyle='-', marker='o', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linestyle='-', marker='o', color='orange', linewidth=2)
    plt.title(f'Training and Validation Loss for MCNN', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Save the plot to the specified path
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    # Close the plot
    plt.close()
