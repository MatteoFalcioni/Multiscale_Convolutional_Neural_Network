from utils.train_data_utils import prepare_dataloader
from scripts.inference import evaluate_model


def evaluate_model(batch_size, full_data_filepath, window_sizes, grid_resolution, features_to_use, num_workers, model, device, model_save_folder, evaluation_data_filepath):
    """
    Evaluates the performance of a trained model on a given dataset and generates a confusion matrix and classification report.

    This function prepares the DataLoader for inference, runs the model evaluation, and prints out the classification report and confusion matrix. It is typically used after the model has been trained to assess its performance on unseen data.

    Args:
        - batch_size (int): The batch size to use for inference.
        - full_data_filepath (str): Filepath to the full dataset.
        - window_sizes (list): List of window sizes to be used for feature images generation.
        - grid_resolution (int): Resolution of the grid used for preparing input data.
        - features_to_use (list): List of features to use during the evaluation (e.g., x, y, z, intensity).
        - num_workers (int): Number of workers to use for loading data in parallel.
        - model (torch.nn.Module): The trained PyTorch model to evaluate.
        - device (torch.device): The device (CPU or GPU) to run the evaluation on.
        - model_save_folder (str): Directory where the model is saved and where the evaluation results will be stored.
        - evaluation_data_filepath (str): Path to the evaluation dataset file.


    Returns:
        None
    """
    print(f'\nEvaluating model performance on file: {evaluation_data_filepath}')

    if evaluation_data_filepath is None:
        raise ValueError(f"You must provide a subset file for evaluation, in order for the model to choose the points from the full dataset\
                         to perform evaluation on.")

    inference_loader, _ = prepare_dataloader(
            batch_size=batch_size,
            data_filepath=full_data_filepath,  
            window_sizes=window_sizes,
            grid_resolution=grid_resolution,
            features_to_use=features_to_use,
            train_split=None,   # prepare the dataloader with the full data for inference (no train/eval split)
            num_workers=num_workers,
            shuffle_train=False,  # we dont want to shuffle data for inference
            subset_file=evaluation_data_filepath    # select points specified by evaluation file to perform evaluation
        )
    
    conf_matrix, class_report = evaluate_model(
        model=model, 
        dataloader=inference_loader, 
        device=device, 
        class_names=['Grass', 'High Vegetation', 'Building', 'Railway', 'Road', 'Car'], 
        model_save_folder=model_save_folder, 
        inference_file_path=evaluation_data_filepath, 
        save=True
    )
    print(f'\nClass report output:\n{class_report}')
    print(f'\nInference process ended.') 