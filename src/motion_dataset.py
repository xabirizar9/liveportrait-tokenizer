import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import os


class MotionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, split: str = 'train', val_split: float = 0.2, seed: int = 42, 
                 compute_stats: bool = True, num_threads: int = 8, device: str = 'cuda'):
        self.data_path = Path(data_path)
        self.pickle_dir = self.data_path / "pickles"
        self.split = split
        self.val_split = val_split
        self.seed = seed
        self.num_threads = num_threads

        # Get all pickle files
        all_pickle_paths = list(self.pickle_dir.glob("*.pkl"))

        # Filter out filenames with reconstruct in the name
        all_pickle_paths = [path for path in all_pickle_paths if "reconstruct" not in path.stem]
        
        # Compute or load statistics for normalization
        self.mean = None
        self.std = None
        
        # Path for cached statistics - use a single file regardless of split
        stats_path = self.data_path / "stats_all.pkl"
        
        if compute_stats:
            # Compute statistics from the entire dataset
            print(f"Computing statistics for entire dataset using {self.num_threads} threads...")
            self.compute_statistics(all_pickle_paths, stats_path)
            print(f"Statistics saved to {stats_path}")
        elif stats_path.exists():
            # Load precomputed statistics if available
            print(f"Loading precomputed statistics from {stats_path}")
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
                self.mean = stats['mean']
                self.std = stats['std']
                print(f"Loaded feature-wise statistics successfully")
        else:
            # Neither computing nor loading statistics
            print(f"Warning: Not computing statistics and no precomputed statistics found at {stats_path}")
            print(f"Normalization will not be applied. Consider setting compute_stats=True")
            
        # Now split the dataset for training/validation
        np.random.seed(seed)
        indices = np.random.permutation(len(all_pickle_paths))
        split_idx = int(len(indices) * (1 - val_split))

        if split == 'train':
            self.pickle_paths = [all_pickle_paths[i] for i in indices[:split_idx]]
        else:  # val
            self.pickle_paths = [all_pickle_paths[i] for i in indices[split_idx:]]
        
        print(f"Loaded {len(self.pickle_paths)} {split} samples")

    def process_pickle_file(self, pickle_path):
        """Process a single pickle file and extract features"""
        features = {
            'kp': [],
            'exp': [],
            'x_s': [],
            't': [],
            'R': [],
            'scale': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
            'kp_velocity': [],
            'exp_velocity': [],
            'kp_acceleration': [],
            'exp_acceleration': [],
            'R_velocity': [],
            'scale_velocity': []
        }
        
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            motion = data['motion']
            c_eyes_lst = data['c_eyes_lst']
            c_lip_lst = data['c_lip_lst']
            fps = data['output_fps']
            
            # Extract features for the current pickle
            current_kp = []
            current_exp = []
            current_R = []
            current_scale = []
            
            # Extract keypoints and expressions for all frames
            for i, m in enumerate(motion):
                kp = torch.tensor(m['kp']).reshape(-1, 63)
                exp = torch.tensor(m['exp']).reshape(-1, 63)
                R = torch.tensor(m['R']).reshape(-1, 9)
                scale = torch.tensor(m['scale']).reshape(-1, 1)
                current_kp.append(kp)
                current_exp.append(exp)
                current_R.append(R)
                current_scale.append(scale)
                
                # Add primary features to the dataset
                features['kp'].append(kp)
                features['exp'].append(exp)
                features['x_s'].append(torch.tensor(m['x_s']).reshape(-1, 63))
                features['t'].append(torch.tensor(m['t']).reshape(-1, 3))
                features['R'].append(torch.tensor(m['R']).reshape(-1, 9))
                features['scale'].append(torch.tensor(m['scale']).reshape(-1, 1))
                features['c_eyes_lst'].append(torch.tensor(c_eyes_lst[i]).reshape(-1, 2))
                features['c_lip_lst'].append(torch.tensor(c_lip_lst[i]).reshape(-1, 1))
            
            # Stack features for this pickle to maintain temporal relationships
            if current_kp:
                stacked_kp = torch.cat(current_kp, dim=0)
                stacked_exp = torch.cat(current_exp, dim=0)
                stacked_R = torch.cat(current_R, dim=0)
                stacked_scale = torch.cat(current_scale, dim=0)
                
                # Calculate derivatives using the correct FPS for this pickle
                # Velocity
                kp_velocity = self.calculate_velocity(stacked_kp, fps)
                exp_velocity = self.calculate_velocity(stacked_exp, fps)
                R_velocity = self.calculate_velocity(stacked_R, fps)
                scale_velocity = self.calculate_velocity(stacked_scale, fps)
                features['kp_velocity'].extend(kp_velocity.unbind(0))
                features['exp_velocity'].extend(exp_velocity.unbind(0))
                features['R_velocity'].extend(R_velocity.unbind(0))
                features['scale_velocity'].extend(scale_velocity.unbind(0))
                
                # Acceleration
                kp_acceleration = self.calculate_acceleration(stacked_kp, fps)
                exp_acceleration = self.calculate_acceleration(stacked_exp, fps)
                features['kp_acceleration'].extend(kp_acceleration.unbind(0))
                features['exp_acceleration'].extend(exp_acceleration.unbind(0))
        except Exception as e:
            print(f"Error processing {pickle_path}: {str(e)}")
        
        return features

    def compute_statistics(self, pickle_paths, stats_path):
        """Compute mean and std of the entire dataset for normalization using threads"""
        print(f"Computing statistics for {len(pickle_paths)} samples using {self.num_threads} threads")
        
        # Initialize combined feature data
        all_features = {
            'kp': [],
            'exp': [],
            'x_s': [],
            't': [],
            'R': [],
            'scale': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
            'kp_velocity': [],
            'exp_velocity': [],
            'kp_acceleration': [],
            'exp_acceleration': [],
            'R_velocity': [],
            'scale_velocity': []
        }
        
        # Process files using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks and get future objects
            future_to_path = {executor.submit(self.process_pickle_file, path): path for path in pickle_paths}
            
            # Process completed futures with a progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(pickle_paths), desc="Processing files"):
                path = future_to_path[future]
                try:
                    # Get the result from the future
                    file_features = future.result()
                    
                    # Add features to the combined data
                    for key in all_features:
                        if file_features[key]:
                            all_features[key].extend(file_features[key])
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
        
        # Stack all features 
        stacked_features = {}
        for key in all_features:
            if all_features[key]:  # Check that list is not empty
                try:
                    stacked_features[key] = torch.stack(all_features[key])
                    print(f"Stacked {len(all_features[key])} values for feature {key}")
                except Exception as e:
                    print(f"Error stacking feature {key}: {str(e)}")
        
        # Compute statistics for each feature type
        self.mean = {}
        self.std = {}
        
        for key in stacked_features:
            self.mean[key] = stacked_features[key].mean(dim=0)
            self.std[key] = stacked_features[key].std(dim=0)
            # Replace zero standard deviations with ones to avoid division by zero
            self.std[key] = torch.where(self.std[key] == 0, torch.ones_like(self.std[key]), self.std[key])
        
        # Save statistics
        with open(stats_path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)
        
        print(f"Computed feature-wise statistics successfully")
    
    def calculate_derivative(self, feature, output_fps, order=1):
        """
        Calculate time derivatives (velocity, acceleration, etc.) for any feature.
        Args:
            feature: Tensor of feature values with shape [frames, ...]
            output_fps: Frames per second for calculating time delta
            order: Order of the derivative (1=velocity, 2=acceleration, etc.)
        Returns:
            Tensor of same shape as input with calculated derivatives
        """
        if order < 1:
            return feature
        
        # Save original shape and flatten to [frames, -1]
        feature_shape = feature.shape
        feature_flat = feature.reshape(feature_shape[0], -1)
        dt = 1 / output_fps
        
        # First derivative calculation (velocity)
        derivative = torch.zeros_like(feature_flat)
        derivative[1:-1] = (feature_flat[2:] - feature_flat[:-2]) / (2 * dt)
        derivative[0] = (feature_flat[1] - feature_flat[0]) / dt
        derivative[-1] = (feature_flat[-1] - feature_flat[-2]) / dt
        
        # For higher order derivatives, recursively call this function
        if order > 1:
            derivative = self.calculate_derivative(derivative.reshape(feature_shape), output_fps, order-1)
            return derivative
            
        return derivative.reshape(feature_shape)
    
    def calculate_velocity(self, feature, output_fps):
        """
        Calculate velocity (first derivative) for any feature.
        Args:
            feature: Tensor of feature values
            output_fps: Frames per second for calculating time delta
        Returns:
            Tensor of same shape as input with calculated velocities
        """
        return self.calculate_derivative(feature, output_fps, order=1)
        
    def calculate_acceleration(self, feature, output_fps):
        """
        Calculate acceleration (second derivative) for any feature.
        Args:
            feature: Tensor of feature values
            output_fps: Frames per second for calculating time delta
        Returns:
            Tensor of same shape as input with calculated accelerations
        """
        return self.calculate_derivative(feature, output_fps, order=2)

    def normalize_features(self, feature, feature_type):
        feature_shape = feature.shape
        feature_flat = feature.reshape(feature_shape[0], -1)
        feature_flat = (feature_flat - self.mean[feature_type]) / self.std[feature_type]
        return feature_flat.reshape(feature_shape)
    
    def denormalize_features(self, feature, feature_type):
        """Denormalize features by applying the reverse normalization operation"""
        feature_shape = feature.shape
        feature_flat = feature.reshape(feature_shape[0], -1)
        feature_flat = feature_flat * self.std[feature_type] + self.mean[feature_type]
        return feature_flat.reshape(feature_shape)

    def denormalize_item(self, item):
        """Denormalize an entire item returned by __getitem__"""
        # Make a copy to avoid modifying the original
        result = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in item.items()}
        
        # Map between item keys and feature types
        feature_map = {
            'kp': 'kp',
            'kp_velocity': 'kp_velocity',
            'kp_acceleration': 'kp_acceleration',
            'exp': 'exp',
            'exp_velocity': 'exp_velocity',
            'exp_acceleration': 'exp_acceleration',
            'x_s': 'x_s',
            't': 't',
            'R': 'R',
            'scale': 'scale',
            'c_eyes_lst': 'c_eyes_lst',
            'c_lip_lst': 'c_lip_lst'
        }
        
        # Denormalize each feature
        for key, feature_type in feature_map.items():
            if key in result and isinstance(result[key], torch.Tensor) and feature_type in self.mean and feature_type in self.std:
                if key == 'scale':
                    result[key] = self.denormalize_features(result[key].reshape(-1, 1), feature_type).squeeze(-1)
                else:
                    result[key] = self.denormalize_features(result[key], feature_type)
        
        return result

    def process_pickle_path(self, pickle_path: str) -> dict:
        """
        Process a single pickle file and return structured features.
        This is the core logic extracted from __getitem__ for reusability.
        
        Args:
            pickle_path: Path to the pickle file to process
            
        Returns:
            Dictionary containing processed and normalized features
        """
        # Load pickle file
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract required data
        n_frames = data['n_frames']
        output_fps = data['output_fps']  # Use the correct FPS for this pickle
        motion = data['motion']
        c_eyes_lst = data['c_eyes_lst']
        c_lip_lst = data['c_lip_lst']
        
        # Extract keypoints for all frames
        kps = torch.stack([torch.tensor(m['kp']) for m in motion])  # Shape: (n_frames, 1, 21, 3)
        exps = torch.stack([torch.tensor(m['exp']) for m in motion])  # Shape: (n_frames, 1, 21, 3)
        x_s = torch.stack([torch.tensor(m['x_s']) for m in motion])  # Shape: (n_frames, 1, 21, 3)
        translations = torch.stack([torch.tensor(m['t']) for m in motion])  # Shape: (n_frames, 1, 3)
        rotations = torch.stack([torch.tensor(m['R']) for m in motion])  # Shape: (n_frames, 3, 3)
        scales = torch.stack([torch.tensor(m['scale']) for m in motion])  # Shape: (n_frames,)
        c_eyes_lst = torch.stack([torch.tensor(c_eyes_lst[i]).squeeze(0) for i in range(len(c_eyes_lst))])  # Shape: (n_frames, 2)
        c_lip_lst = torch.stack([torch.tensor(c_lip_lst[i]).squeeze(0) for i in range(len(c_lip_lst))])  # Shape: (n_frames, 1)    

        # Calculate velocity and acceleration with the correct FPS
        kp_velocity = self.calculate_velocity(kps, output_fps)
        R_velocity = self.calculate_velocity(rotations, output_fps)
        scale_velocity = self.calculate_velocity(scales, output_fps)
        exp_velocity = self.calculate_velocity(exps, output_fps)
        kp_acceleration = self.calculate_acceleration(kps, output_fps)
        exp_acceleration = self.calculate_acceleration(exps, output_fps)
        
        # Normalize primary features
        kps = self.normalize_features(feature=kps, feature_type='kp')
        exps = self.normalize_features(feature=exps, feature_type='exp')
        x_s = self.normalize_features(feature=x_s, feature_type='x_s')
        translations = self.normalize_features(feature=translations, feature_type='t')
        rotations = self.normalize_features(feature=rotations, feature_type='R')
        scales = self.normalize_features(feature=scales, feature_type='scale')
        c_eyes_lst = self.normalize_features(feature=c_eyes_lst, feature_type='c_eyes_lst')
        c_lip_lst = self.normalize_features(feature=c_lip_lst, feature_type='c_lip_lst')

        # Apply statistics-based normalization to derivatives
        kp_velocity = self.normalize_features(feature=kp_velocity, feature_type='kp_velocity')
        kp_acceleration = self.normalize_features(feature=kp_acceleration, feature_type='kp_acceleration')
        exp_velocity = self.normalize_features(feature=exp_velocity, feature_type='exp_velocity')
        exp_acceleration = self.normalize_features(feature=exp_acceleration, feature_type='exp_acceleration')
        R_velocity = self.normalize_features(feature=R_velocity, feature_type='R_velocity')
        scale_velocity = self.normalize_features(feature=scale_velocity, feature_type='scale_velocity')
        
        # Metadata
        metadata = {
            'pickle_path': str(pickle_path),
            'n_frames': n_frames,
            'output_fps': output_fps
        }
        
        # Return the structured data
        return {
            'kp': kps,
            'kp_velocity': kp_velocity,
            'kp_acceleration': kp_acceleration,
            'exp': exps,
            'exp_velocity': exp_velocity, 
            'exp_acceleration': exp_acceleration,
            'x_s': x_s,
            't': translations,
            'R': rotations,
            'R_velocity': R_velocity,
            'scale': scales,
            'scale_velocity': scale_velocity,
            'c_eyes_lst': c_eyes_lst,
            'c_lip_lst': c_lip_lst,
            'metadata': metadata
        }

    def __getitem__(self, idx):
        pickle_path = self.pickle_paths[idx]
        return self.process_pickle_path(str(pickle_path))

    def __len__(self):
        return len(self.pickle_paths)

    def resample_item(self, item, target_fps=None):
        """
        Resample an item (output from __getitem__) to a target framerate.
        
        Args:
            item: Dictionary returned from __getitem__
            target_fps: Target framerate (if None, returns item unchanged)
            
        Returns:
            Resampled item dictionary
        """
        if target_fps is None:
            return item
            
        original_fps = item['metadata']['output_fps']
        original_frames = item['metadata']['n_frames']
        
        if original_fps == target_fps:
            return item
        
        # Calculate new number of frames
        ratio = target_fps / original_fps
        target_frames = max(1, int(original_frames * ratio))
        
        def cubic_interpolate_1d(values, target_length):
            """Vectorized cubic spline interpolation for 1D tensor"""
            if target_length == 1:
                return values[0:1]
            
            original_length = len(values)
            if original_length == 1:
                return values.repeat(target_length)
            
            if original_length == 2:
                # Fall back to linear interpolation for 2 points
                indices = torch.linspace(0, original_length - 1, target_length, device=values.device, dtype=values.dtype)
                floor_indices = torch.floor(indices).long()
                ceil_indices = torch.clamp(floor_indices + 1, max=original_length - 1)
                weights = indices - floor_indices.float()
                return values[floor_indices] * (1 - weights) + values[ceil_indices] * weights
            
            # Cubic spline interpolation for 3+ points using vectorized operations
            device = values.device
            dtype = values.dtype
            
            # Original and target time points
            x_orig = torch.linspace(0, 1, original_length, device=device, dtype=dtype)
            x_new = torch.linspace(0, 1, target_length, device=device, dtype=dtype)
            
            # Compute cubic spline coefficients using natural boundary conditions
            h = x_orig[1:] - x_orig[:-1]  # intervals
            n = original_length
            
            # Build tridiagonal system for second derivatives
            A = torch.zeros(n, n, device=device, dtype=dtype)
            b = torch.zeros(n, device=device, dtype=dtype)
            
            # Natural boundary conditions (second derivative = 0 at endpoints)
            A[0, 0] = 1.0
            A[n-1, n-1] = 1.0
            
            # Interior points - vectorized
            if n > 2:
                interior_indices = torch.arange(1, n-1, device=device)
                A[interior_indices, interior_indices-1] = h[interior_indices-1]
                A[interior_indices, interior_indices] = 2.0 * (h[interior_indices-1] + h[interior_indices])
                A[interior_indices, interior_indices+1] = h[interior_indices]
                
                # Vectorized computation of b values
                dy_right = (values[interior_indices+1] - values[interior_indices]) / h[interior_indices]
                dy_left = (values[interior_indices] - values[interior_indices-1]) / h[interior_indices-1]
                b[interior_indices] = 6.0 * (dy_right - dy_left)
            
            # Solve for second derivatives
            try:
                c = torch.linalg.solve(A, b)
            except:
                # Fallback to linear interpolation if spline fails
                indices = torch.linspace(0, original_length - 1, target_length, device=device, dtype=dtype)
                floor_indices = torch.floor(indices).long()
                ceil_indices = torch.clamp(floor_indices + 1, max=original_length - 1)
                weights = indices - floor_indices.float()
                return values[floor_indices] * (1 - weights) + values[ceil_indices] * weights
            
            # Vectorized spline evaluation
            # Find intervals for all target points at once
            x_new_clamped = torch.clamp(x_new, x_orig[0], x_orig[-1])
            
            # Use searchsorted to find intervals for all points at once
            indices = torch.searchsorted(x_orig[1:], x_new_clamped, right=False)
            indices = torch.clamp(indices, 0, original_length - 2)
            
            # Vectorized cubic spline evaluation
            dx = x_new_clamped - x_orig[indices]
            h_i = h[indices]
            
            # Spline coefficients (vectorized)
            a = values[indices]
            b_coeff = (values[indices + 1] - values[indices]) / h_i - h_i * (2*c[indices] + c[indices + 1]) / 6.0
            c_coeff = c[indices] / 2.0
            d_coeff = (c[indices + 1] - c[indices]) / (6.0 * h_i)
            
            # Evaluate cubic polynomial (vectorized)
            result = a + b_coeff * dx + c_coeff * dx**2 + d_coeff * dx**3
            
            # Handle boundary cases
            left_mask = x_new < x_orig[0]
            right_mask = x_new > x_orig[-1]
            result[left_mask] = values[0]
            result[right_mask] = values[-1]
            
            return result
        
        # Resample each temporal feature
        resampled_item = {}
        temporal_features = ['kp', 'exp', 'x_s', 't', 'R', 'scale', 'c_eyes_lst', 'c_lip_lst']
        
        for key, value in item.items():
            if key == 'metadata':
                # Update metadata
                resampled_item[key] = {
                    'pickle_path': value['pickle_path'],
                    'n_frames': target_frames,
                    'output_fps': target_fps
                }
            elif key in temporal_features and isinstance(value, torch.Tensor):
                # Resample temporal features using cubic spline interpolation
                if value.dim() == 1:
                    # Handle 1D tensors (like scale)
                    resampled_value = cubic_interpolate_1d(value, target_frames)
                elif value.dim() == 2:
                    # Handle 2D tensors (like c_eyes_lst, c_lip_lst, translations)
                    resampled_value = torch.zeros(target_frames, value.shape[1])
                    for i in range(value.shape[1]):
                        resampled_value[:, i] = cubic_interpolate_1d(value[:, i], target_frames)
                elif value.dim() == 3:
                    # Handle 3D tensors (like rotations)
                    resampled_value = torch.zeros(target_frames, value.shape[1], value.shape[2])
                    for i in range(value.shape[1]):
                        for j in range(value.shape[2]):
                            resampled_value[:, i, j] = cubic_interpolate_1d(value[:, i, j], target_frames)
                elif value.dim() == 4:
                    # Handle 4D tensors (like kp, exp, x_s)
                    resampled_value = torch.zeros(target_frames, value.shape[1], value.shape[2], value.shape[3])
                    for i in range(value.shape[1]):
                        for j in range(value.shape[2]):
                            for k in range(value.shape[3]):
                                resampled_value[:, i, j, k] = cubic_interpolate_1d(value[:, i, j, k], target_frames)
                else:
                    raise ValueError(f"Unsupported tensor dimension for feature {key}: {value.dim()}")
                
                resampled_item[key] = resampled_value
            else:
                # Keep non-temporal features as is
                resampled_item[key] = value
        
        # Recalculate derivatives with the new framerate
        if 'kp' in resampled_item:
            resampled_item['kp_velocity'] = self.calculate_velocity(resampled_item['kp'], target_fps)
            resampled_item['kp_acceleration'] = self.calculate_acceleration(resampled_item['kp'], target_fps)
            # Re-normalize derivatives if normalization stats are available
            if self.mean is not None and 'kp_velocity' in self.mean:
                resampled_item['kp_velocity'] = self.normalize_features(resampled_item['kp_velocity'], 'kp_velocity')
                resampled_item['kp_acceleration'] = self.normalize_features(resampled_item['kp_acceleration'], 'kp_acceleration')
        
        if 'exp' in resampled_item:
            resampled_item['exp_velocity'] = self.calculate_velocity(resampled_item['exp'], target_fps)
            resampled_item['exp_acceleration'] = self.calculate_acceleration(resampled_item['exp'], target_fps)
            # Re-normalize derivatives if normalization stats are available
            if self.mean is not None and 'exp_velocity' in self.mean:
                resampled_item['exp_velocity'] = self.normalize_features(resampled_item['exp_velocity'], 'exp_velocity')
                resampled_item['exp_acceleration'] = self.normalize_features(resampled_item['exp_acceleration'], 'exp_acceleration')
        
        if 'R' in resampled_item:
            resampled_item['R_velocity'] = self.calculate_velocity(resampled_item['R'], target_fps)
            if self.mean is not None and 'R_velocity' in self.mean:
                resampled_item['R_velocity'] = self.normalize_features(resampled_item['R_velocity'], 'R_velocity')
        
        if 'scale' in resampled_item:
            resampled_item['scale_velocity'] = self.calculate_velocity(resampled_item['scale'], target_fps)
            if self.mean is not None and 'scale_velocity' in self.mean:
                resampled_item['scale_velocity'] = self.normalize_features(resampled_item['scale_velocity'], 'scale_velocity')
        
        return resampled_item
    

class SnacMotionDataset(MotionDataset):
    def __init__(self, data_path: str, split: str = 'train', val_split: float = 0.2, seed: int = 42, 
                 compute_stats: bool = True, num_threads: int = 8, device: str = 'cuda'):
        super().__init__(data_path, split, val_split, seed, compute_stats, num_threads, device)
        
        self.audio_dir = Path(data_path) / "audios"
        