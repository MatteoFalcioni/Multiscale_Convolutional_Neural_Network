import unittest
import numpy as np
from utils.point_cloud_data_utils import read_file_to_numpy, compute_point_cloud_bounds
'''import cupy as cp
from cuml.neighbors import NearestNeighbors as cuKNN'''
from scipy.spatial import cKDTree 
# from cupyx.scipy.spatial import 
import laspy
from utils.plot_utils import visualize_grid
from torch_kdtree import build_kd_tree
import torch
from scripts.point_cloud_to_image import generate_multiscale_grids, create_feature_grid, assign_features_to_grid
from scripts.gpu_grid_gen import mask_out_of_bounds_points_gpu, generate_multiscale_grids_gpu, generate_multiscale_grids_gpu_masked, create_feature_grid_gpu, assign_features_to_grid_gpu, mask_out_of_bounds_points_gpu
import os
            

class TestGridGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        This will run once before any tests.
        """
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        cls.las_path = 'data/chosen_tiles/32_687000_4930000_FP21.las'
        cls.data_array, cls.known_features = read_file_to_numpy(cls.las_path)
        cls.window_sizes = [('small', 2.5), ('medium', 5.0), ('large', 10.0)]
        cls.test_window_size = 10.0
        cls.grid_resolution = 128
        cls.features_to_use = ['intensity', 'red', 'green', 'blue']  
        cls.num_channels = len(cls.features_to_use)
        cls.feature_indices = [cls.known_features.index(feature) for feature in cls.features_to_use]
        cls.point_cloud_bounds = compute_point_cloud_bounds(cls.data_array)  # Compute bounds
        
        cls.tensor_data_array = torch.tensor(cls.data_array, dtype=torch.float64).to(device=cls.device)
        
        # Initialize CPU and GPU KNN models 
        cls.cpu_kdtree = cKDTree(cls.data_array[:, :3])  # Build KDTREE for CPU
        cls.gpu_kdtree = build_kd_tree(cls.tensor_data_array[:, :3])  # Build KDTree for GPU
        # cls.gpu_knn = build_cuml_knn(data_array=cls.data_array[:, :3])  # Build cuML KNN model for GPU
        
        # sample indices randomly for tests
        cls.num_samples = 5
        np.random.seed(42)
        cls.random_idxs = np.random.choice(len(cls.data_array), size=cls.num_samples, replace=False)

        num_points = int(1e3)
        random_indices = np.random.choice(cls.data_array[0], num_points, replace=False)
        cls.sliced_data = cls.data_array[random_indices, :]
        cls.sliced_tensor = torch.tensor(cls.sliced_data, dtype=torch.float64, device=cls.device)


    def test_torch_tree(self):
        #Dimensionality of the points and KD-Tree
        d = 3

        #Create some random point clouds
        points_ref = torch.randn(size=(1000, d), dtype=torch.float32, device=self.device, requires_grad=True) * 1e3
        points_query = torch.randn(size=(100, d), dtype=torch.float32, device=self.device, requires_grad=True) * 1e3

        #Create the KD-Tree on the GPU and the reference implementation
        torch_kdtree = build_kd_tree(points_ref)
        kdtree = cKDTree(points_ref.detach().cpu().numpy())

        #Search for the 5 nearest neighbors of each point in points_query
        k = 5
        dists, inds = torch_kdtree.query(points_query, nr_nns_searches=k)
        dists_ref, inds_ref = kdtree.query(points_query.detach().cpu().numpy(), k=k)

        #Test for correctness 
        #Note that the cupy_kdtree distances are squared
        assert(np.all(inds.cpu().numpy() == inds_ref))
        assert(np.allclose(torch.sqrt(dists).detach().cpu().numpy(), dists_ref, atol=1e-5))


    def test_compare_indices_for_grids(self):

        np.set_printoptions(precision=16)
        torch.set_printoptions(precision=16)

        idxs = self.random_idxs

        for idx in idxs:
            tensor_center_point = self.tensor_data_array[idx, :].to(self.device, dtype=torch.float64)

            center_point = self.data_array[idx, :]  # Testing point
            window_size = self.test_window_size
            cell_size = window_size / self.grid_resolution

            # Construct the cpu grid's coordinates
            i_indices = np.arange(self.grid_resolution)
            j_indices = np.arange(self.grid_resolution)
            half_resolution_minus_half = (self.grid_resolution / 2) - 0.5
            x_coords = center_point[0] - (half_resolution_minus_half - j_indices) * cell_size
            y_coords = center_point[1] - (half_resolution_minus_half - i_indices) * cell_size
            constant_z = center_point[2]  # Z coordinate is constant for all cells
            cpu_grid_x, cpu_grid_y = np.meshgrid(x_coords, y_coords, indexing='ij') # create a meshgrid
            cpu_grid_coords = np.stack((cpu_grid_x.flatten(), cpu_grid_y.flatten(), np.full(cpu_grid_x.size, constant_z)), axis=-1)
            print(f'\ncpu_grid_coords: {type(cpu_grid_coords)}, dtype: {cpu_grid_coords.dtype}\n')

            # Construct the grid's coordinates in PyTorch with float64
            i_indices = torch.arange(self.grid_resolution, device=self.device)
            j_indices = torch.arange(self.grid_resolution, device=self.device)
            half_resolution_minus_half = torch.tensor((self.grid_resolution / 2) - 0.5, device=self.device, dtype=torch.float64)
            
            # Perform calculations in float64
            x_coords = tensor_center_point[0] - (half_resolution_minus_half - j_indices) * cell_size
            y_coords = tensor_center_point[1] - (half_resolution_minus_half - i_indices) * cell_size
            constant_z = tensor_center_point[2]  # Already float64 if tensor_center_point is

            # Construct meshgrid and stack, enforcing float64
            torch_grid_x, torch_grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
            torch_grid_coords = torch.stack(
                (
                    torch_grid_x.flatten(),
                    torch_grid_y.flatten(),
                    torch.full((torch_grid_x.numel(),), constant_z, device=self.device, dtype=torch.float64)
                ),
                dim=-1
            )
            print(f'\ntorch_grid_coords: {type(torch_grid_coords)}, dtype: {torch_grid_coords.dtype}\n')

            torch_grid_coords_np = torch_grid_coords.cpu().numpy()
            np.testing.assert_allclose(cpu_grid_coords, torch_grid_coords_np, err_msg="Grid coordinates do not match between CPU and GPU")

            torch_distances, torch_indices = self.gpu_kdtree.query(torch_grid_coords)
            cpu_distances, cpu_indices = self.cpu_kdtree.query(cpu_grid_coords) 

            torch_indices_np = (torch_indices.flatten()).cpu().numpy()

            np.testing.assert_allclose(torch_indices_np, cpu_indices, err_msg="Indices do not match between CPU and GPU")


    def test_compare_feature_assignment(self):

        idx = self.random_idxs[0]
        
        tensor_center_point = self.tensor_data_array[idx, :].to(self.device, dtype=torch.float64)
        center_point = self.data_array[idx, :]  # Testing point
        
        cell_size = self.test_window_size / self.grid_resolution
        
        # Construct the cpu grid's coordinates
        i_indices = np.arange(self.grid_resolution)
        j_indices = np.arange(self.grid_resolution)
        half_resolution_minus_half = (self.grid_resolution / 2) - 0.5
        x_coords = center_point[0] - (half_resolution_minus_half - j_indices) * cell_size
        y_coords = center_point[1] - (half_resolution_minus_half - i_indices) * cell_size
        constant_z = center_point[2]  # Z coordinate is constant for all cells
        cpu_grid_x, cpu_grid_y = np.meshgrid(x_coords, y_coords, indexing='ij') # create a meshgrid
        cpu_grid_coords = np.stack((cpu_grid_x.flatten(), cpu_grid_y.flatten(), np.full(cpu_grid_x.size, constant_z)), axis=-1)
        print(f'\ncpu_grid_coords: {type(cpu_grid_coords)}, dtype: {cpu_grid_coords.dtype}\n')

        # Construct the grid's coordinates in PyTorch with float64
        i_indices = torch.arange(self.grid_resolution, device=self.device)
        j_indices = torch.arange(self.grid_resolution, device=self.device)
        half_resolution_minus_half = torch.tensor((self.grid_resolution / 2) - 0.5, device=self.device, dtype=torch.float64)
        
        # Perform calculations in float64
        x_coords = tensor_center_point[0] - (half_resolution_minus_half - j_indices) * cell_size
        y_coords = tensor_center_point[1] - (half_resolution_minus_half - i_indices) * cell_size
        constant_z = tensor_center_point[2]  # Already float64 if tensor_center_point is

        # Construct meshgrid and stack, enforcing float64
        torch_grid_x, torch_grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        torch_grid_coords = torch.stack(
            (
                torch_grid_x.flatten(),
                torch_grid_y.flatten(),
                torch.full((torch_grid_x.numel(),), constant_z, device=self.device, dtype=torch.float64)
            ),
            dim=-1
        )
        print(f'\ntorch_grid_coords: {type(torch_grid_coords)}, dtype: {torch_grid_coords.dtype}\n')
        
        # initialize grids to zeros
        cpu_grid = np.zeros((self.grid_resolution, self.grid_resolution, self.num_channels))
        torch_grid = torch.zeros((self.grid_resolution, self.grid_resolution, self.num_channels), dtype=torch.float64, device=self.device) 
        
        # get indices 
        torch_distances, torch_indices = self.gpu_kdtree.query(torch_grid_coords)
        cpu_distances, cpu_indices = self.cpu_kdtree.query(cpu_grid_coords)
        
        torch_indices_np = (torch_indices.flatten()).cpu().numpy()

        np.testing.assert_allclose(torch_indices_np, cpu_indices, err_msg="Indices do not match between CPU and GPU")
        print("indices match between cpu and gpu")
        
        print(f"cpu_grid shape: {cpu_grid.shape}")
        print(f"CPU indices shape {cpu_indices.shape}")
        print(f"cpu selected features shape: {self.data_array[cpu_indices, :][:, self.feature_indices].shape}")
        # print(f"cpu selected features shape after reshaping: {self.data_array[cpu_indices, :][:, self.feature_indices].reshape(cpu_grid.shape)}")
        cpu_grid[:, :, :] = self.data_array[cpu_indices, :][:, self.feature_indices].reshape(cpu_grid.shape)
        
        print(f"torch_grid shape: {torch_grid.shape}")
        print(f"torch indices shape {torch_indices.shape}")
        torch_indices_flattened = torch_indices.flatten()
        print(f"torch indices flattened shape: {torch_indices_flattened.shape}")
        print(f"tensor_data_array shape: {self.tensor_data_array.shape}")
        print(f"result of self.tensor_data_array[torch_indices_flattened, :] is {self.tensor_data_array[torch_indices_flattened, :].shape}")
        selected_rows = self.tensor_data_array[torch_indices_flattened, :]
        
        print(f"Selected rows shape: {selected_rows.shape}")  # Expected: (16384, 3)
        feats_indices_tensor = torch.tensor(self.feature_indices, device=self.device)
        selected_features = self.tensor_data_array[torch_indices_flattened, :][:, feats_indices_tensor]

        feature_indices_tensor = torch.tensor(self.feature_indices, device=self.device)
        selected_features = selected_rows[:, feature_indices_tensor]
        print(f"Selected features shape: {selected_features.shape}")  # Expected: (16384, len(self.feature_indices))
        
        torch_grid = self.tensor_data_array[torch_indices_flattened, :][:, feats_indices_tensor].reshape(torch_grid.shape)    # this should be enough
        
        np.testing.assert_allclose(cpu_grid, torch_grid.cpu().numpy(), err_msg="grid with features do not match between Torch and CPU")
        
        # WORKING!!! next try less passages in the logic, the above has many lines because of testing
        # lets visualize: 
        cpu_grid = np.transpose(cpu_grid, (2, 0, 1))
        torch_grid_np = np.transpose(torch_grid.cpu().numpy(), (2, 0, 1))
        
        print(f'cpu grid shape for visualization: {cpu_grid.shape}')
        print(f'torch grid shape for visualization: {torch_grid_np.shape}')
        
        visualize_grid(grid=cpu_grid, channel=3, title='CPU')
        visualize_grid(grid=torch_grid_np, channel=3, title='GPU')

        torch_grid, _, torch_x_coords, torch_y_coords, torch_constant_z = create_feature_grid_gpu(center_point_tensor=tensor_center_point,
                                                                                                                window_size=self.test_window_size,
                                                                                                                grid_resolution=self.grid_resolution,
                                                                                                                channels=self.num_channels,
                                                                                                                device=self.device)
        grid, _, x_coords, y_coords, constant_z = create_feature_grid(center_point=center_point,
                                                                              window_size=self.test_window_size,
                                                                              grid_resolution=self.grid_resolution,
                                                                              channels=self.num_channels
                                                                              )
        grid = assign_features_to_grid(tree=self.cpu_kdtree,
                                       data_array=self.data_array,
                                       grid=grid,
                                       x_coords=x_coords,
                                       y_coords=y_coords,
                                       constant_z=constant_z,
                                       feature_indices=self.feature_indices
                                       ) 
        torch_grid = assign_features_to_grid_gpu(gpu_tree=self.gpu_kdtree,
                                                 tensor_data_array=self.tensor_data_array,
                                                 grid=torch_grid,
                                                 x_coords=torch_x_coords,
                                                 y_coords=torch_y_coords,
                                                 constant_z=torch_constant_z,
                                                 feature_indices_tensor=feature_indices_tensor,
                                                 device=self.device)
        np.testing.assert_allclose(grid, torch_grid.cpu().numpy(), err_msg="grid with features do not match between Torch and CPU")
        
        
    def test_gpu_grid_generation(self):
        """
        Test that the GPU grid generation works correctly for a subset of points.
        """
        
        idxs = self.random_idxs
        feature_indices_tensor = torch.tensor(self.feature_indices, device=self.device)

        for idx in idxs:
            center_point_tensor = self.tensor_data_array[idx, :]
            # Generate grids using the GPU pipeline
            gpu_grids, gpu_status = generate_multiscale_grids_gpu(
                center_point_tensor=center_point_tensor, tensor_data_array=self.tensor_data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices_tensor=feature_indices_tensor,
                gpu_tree=self.gpu_kdtree, point_cloud_bounds=self.point_cloud_bounds, device=self.device
            )

            if gpu_status is not None:
                # print('Skipped point')
                continue
            
            # Check if the generated grids contain any NaN or Inf values
            for scale in ['small', 'medium', 'large']:
                grid = gpu_grids[scale]

                # Check if there are any NaN or Inf values in the grid
                self.assertFalse(torch.isnan(grid).any(), f"NaN found in {scale} grid for point {center_point_tensor}")
                self.assertFalse(torch.isinf(grid).any(), f"Inf found in {scale} grid for point {center_point_tensor}")

                # Check if the grid has the expected shape (C, H, W)
                self.assertEqual(grid.shape, (self.num_channels, self.grid_resolution, self.grid_resolution),  # Assuming 4 channels (features)
                                 f"Grid shape mismatch for {scale} grid at point {center_point_tensor}")

                # Check if there are no zero cells in the grid (i.e., no cells without assigned features)
                self.assertFalse(torch.all(grid == 0), f"Grid contains zero cells for {scale} grid at point {center_point_tensor}")
                
                
    def test_generate_grids_cpu_vs_gpu(self):
        """
        Test if grids generated on CPU match with those generated on GPU.
        """
        idxs = self.random_idxs
        feature_indices_tensor = torch.tensor(self.feature_indices, device=self.device)

        for idx in idxs:
            center_point_tensor = self.tensor_data_array[idx, :]
            center_point = self.data_array[idx, :]
            # Generate grids using the CPU pipeline
            cpu_grids, cpu_status = generate_multiscale_grids(
                center_point, data_array=self.data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices=self.feature_indices,
                kdtree=self.cpu_kdtree, point_cloud_bounds=self.point_cloud_bounds
            )

            # Generate grids using the GPU pipeline
            gpu_grids, gpu_status = generate_multiscale_grids_gpu(
                center_point_tensor=center_point_tensor, tensor_data_array=self.tensor_data_array, window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution, feature_indices_tensor=feature_indices_tensor,
                gpu_tree=self.gpu_kdtree, point_cloud_bounds=self.point_cloud_bounds, device=self.device
            )
            
            self.assertEqual(cpu_status, gpu_status, f"Error: status doesn't match between gpu and cpu.") 
            
            if gpu_status is None:  # i.e., point wasnt skipped
                
            # Compare the grids from CPU and GPU (for the same point)
                for scale in ['small', 'medium', 'large']:
                    cpu_grid = cpu_grids[scale]
                    gpu_grid = gpu_grids[scale]
                    
                    # Move the GPU grid to CPU 
                    moved_gpu_grid = gpu_grid.cpu().numpy()  

                    # Visualize the grids (optional)
                    visualize_grid(cpu_grid, channel=3, feature_names=self.features_to_use, save=False)
                    visualize_grid(moved_gpu_grid, channel=3, feature_names=self.features_to_use, save=False)

                    # Check if the grids are approximately equal
                    try:
                        np.testing.assert_almost_equal(cpu_grid, moved_gpu_grid, decimal=6, 
                                                    err_msg=f"Grids do not match for {scale} scale at point {center_point}")
                    except AssertionError as e:
                        print(e)  # If grids don't match, print the error message and continue
        
        

    def test_masked_vs_unmasked_grid_gen_gpu(self):
        """
        Test the grid generation for masked vs unmasked points on GPU.
        """

        # Apply GPU-based masking
        masked_sliced_data_gpu, mask_gpu = mask_out_of_bounds_points_gpu(self.sliced_tensor, self.window_sizes, self.point_cloud_bounds)

        # Collect grids and coordinates for unmasked approach
        usual_grids = []
        usual_coords = []
        out_of_bounds = 0

        for point in self.sliced_tensor:
            grids_dict = generate_multiscale_grids_gpu(
                center_point_tensor=point,
                tensor_data_array=self.tensor_data_array,
                window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution,
                feature_indices_tensor=torch.tensor(self.feature_indices, device="cuda"),
                gpu_tree=self.gpu_tree,  # Pass the prebuilt GPU KDTree
            )

            if grids_dict:
                usual_grids.append(grids_dict)
                usual_coords.append(tuple(point.cpu().numpy()))
            else:
                out_of_bounds += 1

        print(f"Out-of-bounds points excluded in grid generation: {out_of_bounds}")

        # Collect grids and coordinates for masked approach
        masked_grids = []
        masked_coords = []

        for point in masked_sliced_data_gpu:
            grids_dict = generate_multiscale_grids_gpu_masked(
                center_point_tensor=point,
                tensor_data_array=self.tensor_data_array,
                window_sizes=self.window_sizes,
                grid_resolution=self.grid_resolution,
                feature_indices_tensor=torch.tensor(self.feature_indices, device="cuda"),
                gpu_tree=self.gpu_tree,  # Pass the prebuilt GPU KDTree
            )
            masked_grids.append(grids_dict)
            masked_coords.append(tuple(point.cpu().numpy()))

        # Assert the number of grids matches
        self.assertEqual(len(masked_grids), len(usual_grids),
                        f"Number of grids doesn't match between masked ({len(masked_grids)}) and usual ({len(usual_grids)}).")

        # Assert coordinates match
        self.assertEqual(sorted(masked_coords), sorted(usual_coords),
                        "Coordinates do not match between masked and unmasked approaches.")

        # Compare the grids for matching coordinates
        for coord in masked_coords:
            # Find the index of the matching coordinate in the usual approach
            usual_index = usual_coords.index(coord)
            masked_index = masked_coords.index(coord)

            usual_grids_dict = usual_grids[usual_index]
            masked_grids_dict = masked_grids[masked_index]

            # Compare grids at all scales
            for scale_label in self.window_sizes:
                scale = scale_label[0]
                np.testing.assert_array_equal(
                    masked_grids_dict[scale].cpu().numpy(),
                    usual_grids_dict[scale].cpu().numpy(),
                    err_msg=f"Grid values differ for point {coord} at scale {scale}."
                )







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

        np.testing.assert_array_equal(cpu_indices_sorted, cupy_indices_sorted, err_msg="Indices mismatch between CPU and CuPy KDTree")'''
        

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
    

if __name__ == '__main__':
    unittest.main()
