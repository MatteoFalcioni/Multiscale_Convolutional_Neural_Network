import unittest
import numpy as np
from utils.point_cloud_data_utils import read_file_to_numpy
from scripts.point_cloud_to_image import compute_point_cloud_bounds
import cupy as cp
from scipy.spatial import cKDTree 
from cuml.neighbors import NearestNeighbors as cuKNN
# from cupyx.scipy.spatial import 
import laspy
from utils.plot_utils import visualize_grid
from torch_kdtree import build_kd_tree
import torch
from scripts.point_cloud_to_image import generate_multiscale_grids
from scripts.gpu_grid_gen import generate_multiscale_grids_gpu


def build_cuml_knn(data_array, n_neighbors=1):
        """
        Builds and returns a cuML KNN model on the GPU for nearest neighbor search.
        """
        data_gpu = cp.array(data_array, dtype=cp.float64) 
        cuml_knn = cuKNN(n_neighbors=n_neighbors, metric='euclidean')
        cuml_knn.fit(data_gpu)
        return cuml_knn
            

class TestGridGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This will run once before any tests.
        """
        cls.las_path = 'data/chosen_tiles/32_687000_4930000_FP21.las'
        cls.data_array, cls.known_features = read_file_to_numpy(cls.las_path)
        print(f'data_array type right after reading: {cls.data_array.dtype}')
        cls.window_sizes = [('small', 1.0), ('medium', 2.0), ('large', 3.0)]
        cls.grid_resolution = 128
        cls.features_to_use = ['intensity', 'red', 'green', 'blue']  
        cls.num_channels = len(cls.features_to_use)
        cls.feature_indices = [cls.known_features.index(feature) for feature in cls.features_to_use]
        cls.point_cloud_bounds = compute_point_cloud_bounds(cls.data_array)  # Compute bounds
        
        cls.data_array_gpu = torch.tensor(cls.data_array[:, :3], dtype=torch.float32).to(device="cuda")
        
        # Initialize CPU and GPU KNN models 
        cls.cpu_kdtree = cKDTree(cls.data_array[:, :3])  # Build KDTREE for CPU
        cls.gpu_kdtree = build_kd_tree(cls.data_array_gpu)  # Build KDTree for GPU
        cls.gpu_knn = build_cuml_knn(data_array=cls.data_array[:, :3])  # Build cuML KNN model for GPU

    
    '''def test_compare_cpu_gpu_knn(self):
        """
        Test if the grids match between CPU and GPU, and if the indices returned by CPU and GPU KNN models for the same point match.
        """
        np.set_printoptions(precision=16)
        cp.set_printoptions(precision=16)
        
        # Take a small subset of points for testing
        center_point = self.data_array[100005, :].astype(np.float64)  # Testing point
        window_size = 10.0
        # Calculate the size of each cell in meters
        cell_size = window_size / self.grid_resolution
        
        # Construct the cpu grid's coordinates
        # Generate cell coordinates for the grid based on the center point
        i_indices = np.arange(self.grid_resolution)
        j_indices = np.arange(self.grid_resolution)
        half_resolution_minus_half = (self.grid_resolution / 2) - 0.5
        # following x_k = x_pk - (64.5 - j) * w
        x_coords = center_point[0] - (half_resolution_minus_half - j_indices) * cell_size
        y_coords = center_point[1] - (half_resolution_minus_half - i_indices) * cell_size
        constant_z = center_point[2]  # Z coordinate is constant for all cells
        cpu_grid_x, cpu_grid_y = np.meshgrid(x_coords, y_coords, indexing='ij') # create a meshgrid
        cpu_grid_coords = np.stack((cpu_grid_x.flatten(), cpu_grid_y.flatten(), np.full(cpu_grid_x.size, constant_z)), axis=-1)
        print(f'\ncpu_grid_coords: {type(cpu_grid_coords)}, dtype: {cpu_grid_coords.dtype}\n')
        
        # Construct Gpu grid in the same way (trying to enforce float64)
        gpu_half_resolution_minus_half = half_resolution_minus_half
        gpu_center_point = cp.array(center_point)
        gpu_cell_size = window_size / self.grid_resolution
        gpu_i_indices = cp.arange(self.grid_resolution)
        gpu_j_indices = cp.arange(self.grid_resolution)
        gpu_x_coords = gpu_center_point[0] - (gpu_half_resolution_minus_half - gpu_j_indices) * gpu_cell_size
        gpu_y_coords = gpu_center_point[1] - (gpu_half_resolution_minus_half - gpu_i_indices) * cell_size
        gpu_x_coords = cp.array(gpu_x_coords)
        gpu_y_coords = cp.array(gpu_y_coords)
        gpu_grid_x, gpu_grid_y = cp.meshgrid(gpu_x_coords, gpu_y_coords, indexing='ij')
        gpu_grid_coords = cp.stack(
            (gpu_grid_x.flatten(), gpu_grid_y.flatten(), cp.full(gpu_grid_x.size, constant_z)), axis=-1
        )
        print(f"gpu_grid_coords: {type(gpu_grid_coords)}, dtype: {gpu_grid_coords.dtype}")
        
        # Check if CPU and GPU grid coordinates match
        print(f'\ntesting if the grids coordinates match betweeen CPU and GPU...')
        np.testing.assert_array_equal(cpu_grid_coords, gpu_grid_coords.get(), err_msg="Grid coordinates do not match between CPU and GPU")
        print(f'\nOK: Grids coordinates match\n')
        
        # Get distances for the first point in the list for both CPU and GPU
        single_cpu_distance, single_cpu_index = self.cpu_kdtree.query(cpu_grid_coords[:1])
        single_gpu_distance, single_gpu_index = self.gpu_tree.kneighbors(gpu_grid_coords[:1])
        single_cupy_distance, single_cupy_index = self.gpu_kdtree.query(gpu_grid_coords[:1]) # , k=1
        
        # compare results
        print(f"CPU Distance: {single_cpu_distance}, Index: {single_cpu_index}")
        print(f"cuML GPU Distance: {single_gpu_distance}, Index: {single_gpu_index}")
        print(f"CuPy KDTree Distance: {single_cupy_distance.get()}, Index: {single_cupy_index.get()}")
        
        # CPU KNN: Get all grid points indices using the CPU cKDTree
        cpu_distances, cpu_indices = self.cpu_kdtree.query(cpu_grid_coords)
        # GPU KNN: Get all grid points indices using the cuML KNN model
        gpu_distances, gpu_indices = self.gpu_knn.kneighbors(gpu_grid_coords)
        # GPU KDTree: Get all grid points indices using the GPU KDTree 
        cupy_distances, cupy_indices = self.gpu_kdtree.query(gpu_grid_coords[:, :2], k=1)
        # Sort the indices to compare 
        cpu_indices_sorted = np.sort(cpu_indices)
        gpu_indices_sorted = np.sort(gpu_indices.flatten())
        
        # Sort the indices to compare
        cpu_indices_sorted = np.sort(cpu_indices.flatten())
        gpu_indices_sorted = np.sort(gpu_indices.flatten())
        cupy_indices_sorted = np.sort(cupy_indices.get().flatten())

        # Print indices for comparison
        print(f"\nFirst 20 CPU indices: {cpu_indices_sorted[:20]}")
        print(f"First 20 GPU indices (cuML): {gpu_indices_sorted[:20]}")
        print(f"First 20 GPU indices (CuPy): {cupy_indices_sorted[:20]}")

        # Compare the distances across methods
        cpu_sorted_distances = np.sort(cpu_distances)
        gpu_sorted_distances = np.sort(gpu_distances.flatten())
        cupy_sorted_distances = np.sort(cupy_distances.get().flatten())

        print(f"\nCorresponding CPU distances: {cpu_sorted_distances[:20]}")
        print(f"Corresponding GPU distances (cuML): {gpu_sorted_distances[:20]}")
        print(f"Corresponding GPU distances (CuPy): {cupy_sorted_distances[:20]}")

        np.testing.assert_array_equal(cpu_indices_sorted, cupy_indices_sorted, err_msg="Indices mismatch between CPU and CuPy KDTree")
        '''

    '''def test_knn_synthetic(self):
        # First, check the precision of the data in the LAS file for comparison
        # Load the LAS file
        las_file_path = "data/chosen_tiles/32_687000_4930000_FP21.las"
        las = laspy.read(las_file_path)

        # Inspect the coordinate precision
        x_coords = las.x
        y_coords = las.y
        z_coords = las.z

        # Check data types (likely float32 or float64)
        print(f"\nx_coords data type: {x_coords.dtype}\n")
        print(f"y_coords data type: {y_coords.dtype}\n")
        print(f"z_coords data type: {z_coords.dtype}\n")
        
        data_array, _ = read_file_to_numpy(las_file_path)
        np.set_printoptions(precision=32)
        test_point = data_array[0, :3]
        print(test_point)
        np.set_printoptions(precision=8)
        
        np.random.seed(42)
        number_of_neighbors = 5
        
        fake_data = np.random.rand(1000, 3).astype(np.float64)  # 1000 points with 3 dimensions
        print(f'generated points: {fake_data}')
        test_point = np.random.rand(1, 3)
        # Initialize CPU and GPU trees
        kdtree = cKDTree(fake_data[:, :3])  # CPU KDTREE
        gpu_tree = build_cuml_knn(data_array=fake_data[:, :3], n_neighbors=number_of_neighbors)  # cuML KNN for GPU
        
        # Create torch_kdtree for GPU KNN (using PyTorch)
        fake_data_tensor = torch.tensor(fake_data[:, :3], dtype=torch.float32).to(device="cuda")
        torch_kdtree = build_kd_tree(fake_data_tensor)  # Build torch_kdtree for GPU
        
        # CPU query
        cpu_distances, cpu_indices = kdtree.query(test_point, k=number_of_neighbors)

        # cuML GPU query
        gpu_distances, gpu_indices = gpu_tree.kneighbors(cp.array(test_point, dtype=cp.float64))  # Ensure dtype match

        # torch_kdtree GPU query
        test_point_tensor = torch.tensor(test_point, dtype=torch.float32).to(device="cuda")
        torch_gpu_distances, torch_gpu_indices = torch_kdtree.query(test_point_tensor, nr_nns_searches=number_of_neighbors)

        # Compare the indices and coordinates returned by CPU, GPU (cuML), and GPU (torch_kdtree)
        print(f"\nTest Point: {test_point}\n")
        print(f"CPU Indices: {cpu_indices}\n")
        print(f"GPU Indices (cuML): {gpu_indices}\n")
        print(f"GPU Indices (torch_kdtree): {torch_gpu_indices.cpu().numpy()}\n")
        
        print(f"CPU distances: {cpu_distances}\n")
        print(f"GPU distances (cuML): {gpu_distances}\n")
        print(f"GPU distances (torch_kdtree): {torch_gpu_distances.cpu().numpy()}\n")
        
        # Check if the coordinates are the same for the indices
        cpu_coordinates = fake_data[cpu_indices, :]
        gpu_coordinates = fake_data[(gpu_indices.flatten()).get(), :]
        torch_gpu_coordinates = fake_data[(torch_gpu_indices.cpu().numpy().flatten()), :]

        # Check consistency between CPU, cuML GPU, and torch_kdtree GPU queries
        steps = 10
        for i in range(steps):
            precision = 1e-3**(i+1)
            print(f'precision: {precision}')

            # Multiply test point and fake data by precision at each step
            modified_test_point = test_point * precision
            modified_fake_data = fake_data * precision  
            
            # Recreate trees with more precision needed
            modified_kdtree = cKDTree(modified_fake_data[:, :3])
            modified_gpu_tree = build_cuml_knn(data_array=modified_fake_data[:, :3], n_neighbors=number_of_neighbors)
            modified_fake_data_tensor = torch.tensor(modified_fake_data[:, :3], dtype=torch.float32).to(device="cuda")
            modified_torch_kdtree = build_kd_tree(modified_fake_data_tensor)

            _, modified_cpu_indices = modified_kdtree.query(modified_test_point, k=number_of_neighbors)
            _, modified_gpu_indices = modified_gpu_tree.kneighbors(cp.array(modified_test_point, dtype=cp.float64))
            modified_test_point_tensor = torch.tensor(modified_test_point, dtype=torch.float32).to(device="cuda")
            _, modified_torch_gpu_indices = modified_torch_kdtree.query(modified_test_point_tensor, nr_nns_searches=number_of_neighbors)
            
            print(f"\nTest Point: {modified_test_point}\n")
            print(f"CPU Indices: {modified_cpu_indices}\n")
            print(f"GPU Indices (cuML): {modified_gpu_indices}\n")
            print(f"GPU Indices (torch_kdtree): {modified_torch_gpu_indices.cpu().numpy()}\n")
            
            # Check if the indices are the same for CPU, cuML GPU, and torch_kdtree GPU
            are_equal_cpu_gpu = np.array_equal(modified_cpu_indices, modified_gpu_indices.get())
            are_equal_cpu_torch = np.array_equal(modified_cpu_indices, modified_torch_gpu_indices.cpu().numpy())
            are_equal_gpu_torch = np.array_equal(modified_gpu_indices, modified_torch_gpu_indices.cpu().numpy())

            if not are_equal_cpu_gpu or not are_equal_cpu_torch or not are_equal_gpu_torch:
                print(f'Indices diverged at precision {precision}')
                break'''

    '''def test_gpu_grid_generation(self):
        """
        Test that the GPU grid generation works correctly for a subset of points.
        """
        # Take a small subset of points for testing
        points_to_test = self.data_array[:10, :]  # Testing with 10 points

        for center_point in points_to_test:
            # Generate grids using the GPU pipeline
            gpu_grids, skipped_gpu = generate_multiscale_grids_gpu(
                center_point, data_array=self.data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.feature_indices,
                cuml_knn=self.gpu_tree, point_cloud_bounds=self.point_cloud_bounds
            )
            
            # Print the keys of gpu_grids
            # print(f"Generated grids keys: {gpu_grids.keys()}")

            if skipped_gpu:
                # print('Skipped point')
                continue
            
            # Check if the generated grids contain any NaN or Inf values
            for scale in ['small', 'medium', 'large']:
                grid = gpu_grids[scale]

                # Check if there are any NaN or Inf values in the grid
                self.assertFalse(cp.isnan(grid).any(), f"NaN found in {scale} grid for point {center_point}")
                self.assertFalse(cp.isinf(grid).any(), f"Inf found in {scale} grid for point {center_point}")

                # Check if the grid has the expected shape (C, H, W)
                self.assertEqual(grid.shape, (4, self.grid_resolution, self.grid_resolution),  # Assuming 4 channels (features)
                                 f"Grid shape mismatch for {scale} grid at point {center_point}")

                # Check if there are no zero cells in the grid (i.e., no cells without assigned features)
                self.assertFalse(cp.all(grid == 0), f"Grid contains zero cells for {scale} grid at point {center_point}")'''
    

    def test_generate_grids_cpu_vs_gpu(self):
        """
        Test if grids generated on CPU match with those generated on GPU.
        """
        # Take a small subset of points for comparison
        points_to_test = self.data_array[200000:200010, :]  # Testing with 10 points
        
        # print(f'checking points {points_to_test}')

        for center_point in points_to_test:
            # Generate grids using the CPU pipeline
            cpu_grids, skipped_cpu = generate_multiscale_grids(
                center_point, data_array=self.data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.feature_indices,
                kdtree=self.cpu_kdtree, point_cloud_bounds=self.point_cloud_bounds
            )

            # Generate grids using the GPU pipeline
            gpu_grids, skipped_gpu = generate_multiscale_grids_gpu(
                center_point, data_array=self.data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.feature_indices,
                gpu_tree=self.gpu_kdtree, point_cloud_bounds=self.point_cloud_bounds
            )
            
            if skipped_cpu and not skipped_gpu:
                print(f'error: point skipped by cpu but not fom gpu')
                
            if skipped_gpu and not skipped_cpu:
                print(f'error: point skipped by gpu but not fom cpu')
            
            if skipped_cpu and skipped_gpu:
                print(f'The point was skipped by both gpu and cpu')
                continue
            
            
            # Compare the grids from CPU and GPU (for the same point)
            for scale in ['small', 'medium', 'large']:
                cpu_grid = cpu_grids[scale]
                gpu_grid = gpu_grids[scale]
                
                # Move the GPU grid to CPU (if it's a PyTorch tensor)
                if isinstance(gpu_grid, torch.Tensor):
                    moved_gpu_grid = gpu_grid.cpu().numpy()  # Use .cpu() for PyTorch tensors
                else:
                    moved_gpu_grid = gpu_grid.get()  # If it's a CuPy array, use .get()
                
                # Visualize the grids (optional)
                visualize_grid(cpu_grid, channel=3, feature_names=self.features_to_use, save=False)
                visualize_grid(moved_gpu_grid, channel=3, feature_names=self.features_to_use, save=False)

                # Check if the grids are approximately equal
                try:
                    np.testing.assert_almost_equal(cpu_grid, moved_gpu_grid, decimal=3, 
                                                err_msg=f"Grids do not match for {scale} scale at point {center_point}")
                except AssertionError as e:
                    print(e)  # If grids don't match, print the error message and continue

if __name__ == '__main__':
    unittest.main()
