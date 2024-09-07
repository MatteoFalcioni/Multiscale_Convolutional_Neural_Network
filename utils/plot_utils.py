import matplotlib.pyplot as plt


def plot_loss(train_losses, val_losses, save_path=None):
    """
    Plots the training and validation loss over epochs.

    Args:
    - train_losses (list of float): List containing the training loss values for each epoch.
    - val_losses (list of float): List containing the validation loss values for each epoch.
    - save_path (str, optional): If provided, saves the plot to the given path. Otherwise, it shows the plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot training and validation loss
    plt.plot(train_losses, label='Training Loss', linestyle='-', marker='o', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linestyle='-', marker='o', color='red', linewidth=2)

    # Add a title and labels
    plt.title('Training and Validation Loss over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Add a grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Display or save the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()

