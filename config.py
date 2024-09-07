import torch

# Hyperparameters
BATCH_SIZE = 16  # Not explicitly mentioned in the article
EPOCHS = 20
LEARNING_RATE = 0.01  # Initial learning rate as mentioned in the (newest) article
LEARNING_RATE_DECAY_EPOCHS = 5  # Halve learning rate every 5 epochs (mentioned)
LOSS_FUNCTION = 'CrossEntropy'  # Using CrossEntropy

# Optimizer
OPTIMIZER = 'SGD'  # Stochastic Gradient Descent

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model save path base directory
MODEL_SAVE_DIR = "models/saved/"

