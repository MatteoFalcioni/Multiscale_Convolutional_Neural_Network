import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from scripts.evaluate_model import evaluate_model
from utils.train_data_utils import load_parameters, load_model
from utils.point_cloud_data_utils import read_file_to_numpy


class TestEvaluationProcess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Run once for the entire test class
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.full_data_filepath = 'data/datasets/sampled_full_dataset/sampled_data_5251680.csv'
        cls.evaluation_data_filepath = 'data/datasets/eval_dataset.csv'
        
        full_data_array, full_features = read_file_to_numpy(cls.full_data_filepath)
        evaluation_array, evaluation_features = read_file_to_numpy(cls.evaluation_data_filepath)
        print(f"full data array shape: {full_data_array.shape}\nFull data array features: {full_features}\n")
        print(f"Evalaution data array shape: {evaluation_array.shape}\nEvaluation array features: {evaluation_features}\n")
        
        # load model 
        loaded_model_path = 'models/saved/mcnn_model_20241116_143003/model.pth'
        cls.loaded_features, cls.num_loaded_channels, cls.loaded_window_sizes = load_parameters(loaded_model_path)
        cls.loaded_model = load_model(model_path=loaded_model_path, device=cls.device, num_channels=cls.num_loaded_channels)
        
        cls.criterion = nn.CrossEntropyLoss()
        cls.optimizer = optim.SGD(cls.loaded_model.parameters(), lr=0.01)
        cls.scheduler = optim.lr_scheduler.StepLR(cls.optimizer, step_size=1, gamma=0.5)
        cls.grid_resolution = 128
        cls.batch_size = 16
        cls.num_workers = 32
        
        cls.test_folder = 'tests/test_evaluation'
        
        
    def test_evaluation(self):
        evaluate_model(batch_size=self.batch_size, 
                       full_data_filepath=self.full_data_filepath, 
                       window_sizes=self.loaded_window_sizes, 
                       grid_resolution=self.grid_resolution, 
                       features_to_use=self.loaded_features, 
                       num_workers=self.num_workers, 
                       model=self.loaded_model, 
                       device=self.device, 
                       model_save_folder=self.test_folder, 
                       evaluation_data_filepath=self.evaluation_data_filepath)

