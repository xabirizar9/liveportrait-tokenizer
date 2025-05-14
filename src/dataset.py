import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, split: str = 'train', val_split: float = 0.2, seed: int = 42, 
                 compute_stats: bool = True):
        self.data_path = Path(data_path)
        self.pickle_dir = self.data_path / "pickles"
        self.split = split
        self.val_split = val_split
        self.seed = seed

        # Get all pickle files
        all_pickle_paths = list(self.pickle_dir.glob("*.pkl"))

        # Filter filenames with reconstruct in the name
        all_pickle_paths = [path for path in all_pickle_paths if "reconstruct" not in path.stem]
        
        # Compute or load statistics for normalization
        self.mean = None
        self.std = None
        
        # Path for cached statistics - use a single file regardless of split
        stats_path = self.data_path / "stats_all.pkl"
        
        if compute_stats:
                # Compute statistics from the entire dataset
                print(f"Computing statistics for entire dataset...")
                self.compute_statistics(all_pickle_paths, stats_path)
                print(f"Statistics saved to {stats_path}")
        else:
            if stats_path.exists():
                print(f"Loading precomputed statistics from {stats_path}")
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                    self.mean = stats['mean']
                    self.std = stats['std']
                    print(f"Loaded feature-wise statistics successfully")
            
        # Now split the dataset for training/validation
        np.random.seed(seed)
        indices = np.random.permutation(len(all_pickle_paths))
        split_idx = int(len(indices) * (1 - val_split))

        if split == 'train':
            self.pickle_paths = [all_pickle_paths[i] for i in indices[:split_idx]]
        else:  # val
            self.pickle_paths = [all_pickle_paths[i] for i in indices[split_idx:]]
        
        print(f"Loaded {len(self.pickle_paths)} {split} samples")

    def compute_statistics(self, pickle_paths, stats_path):
        """Compute mean and std of the entire dataset for normalization"""
        feature_data = {
            'kp': [],
            'exp': [],
            'x_s': [],
            't': [],
            'R': [],
            'scale': [],
            'c_eyes': [],
            'c_lip': [],
            'kp_velocity': [],
            'exp_velocity': [],
            'kp_acceleration': [],
            'exp_acceleration': []
        }

        print(f"Computing statistics for {len(pickle_paths)} samples")
        
        # Collect features and calculate derivatives with correct FPS for each pickle
        for pickle_path in pickle_paths:    
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            motion = data['motion']
            c_eyes_lst = data['c_eyes_lst']
            c_lip_lst = data['c_lip_lst']
            fps = data['output_fps']  # Get specific FPS for this pickle
            
            # Extract features for the current pickle
            current_kp = []
            current_exp = []
            
            # Extract keypoints and expressions for all frames
            for i, m in enumerate(motion):
                kp = torch.tensor(m['kp']).reshape(-1, 63)
                exp = torch.tensor(m['exp']).reshape(-1, 63)
                
                current_kp.append(kp)
                current_exp.append(exp)
                
                # Add primary features to the dataset
                feature_data['kp'].append(kp)
                feature_data['exp'].append(exp)
                feature_data['x_s'].append(torch.tensor(m['x_s']).reshape(-1, 63))
                feature_data['t'].append(torch.tensor(m['t']).reshape(-1, 3))
                feature_data['R'].append(torch.tensor(m['R']).reshape(-1, 9))
                feature_data['scale'].append(torch.tensor(m['scale']).reshape(-1, 1))
                feature_data['c_eyes'].append(torch.tensor(c_eyes_lst[i]).reshape(-1, 2))
                feature_data['c_lip'].append(torch.tensor(c_lip_lst[i]).reshape(-1, 1))
            
            # Stack features for this pickle to maintain temporal relationships
            stacked_kp = torch.cat(current_kp, dim=0)
            stacked_exp = torch.cat(current_exp, dim=0)
            
            # Calculate derivatives using the correct FPS for this pickle
            # Velocity
            kp_velocity = self.calculate_velocity(stacked_kp, fps)
            exp_velocity = self.calculate_velocity(stacked_exp, fps)
            feature_data['kp_velocity'].extend(kp_velocity.unbind(0))
            feature_data['exp_velocity'].extend(exp_velocity.unbind(0))
            
            # Acceleration
            kp_acceleration = self.calculate_acceleration(stacked_kp, fps)
            exp_acceleration = self.calculate_acceleration(stacked_exp, fps)
            feature_data['kp_acceleration'].extend(kp_acceleration.unbind(0))
            feature_data['exp_acceleration'].extend(exp_acceleration.unbind(0))
        
        # Stack all features 
        stacked_features = {}
        for key in feature_data:
            if feature_data[key]:  # Check that list is not empty
                stacked_features[key] = torch.stack(feature_data[key])
        
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
            'c_eyes': 'c_eyes',
            'c_lip': 'c_lip'
        }
        
        # Denormalize each feature
        for key, feature_type in feature_map.items():
            if key in result and isinstance(result[key], torch.Tensor) and feature_type in self.mean and feature_type in self.std:
                if key == 'scale':
                    result[key] = self.denormalize_features(result[key].reshape(-1, 1), feature_type).squeeze(-1)
                else:
                    result[key] = self.denormalize_features(result[key], feature_type)
        
        return result

    def __getitem__(self, idx):
        pickle_path = self.pickle_paths[idx]
        
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
        c_eyes = torch.stack([torch.tensor(c_eyes_lst[i]).squeeze(0) for i in range(len(c_eyes_lst))])  # Shape: (n_frames, 2)
        c_lip = torch.stack([torch.tensor(c_lip_lst[i]).squeeze(0) for i in range(len(c_lip_lst))])  # Shape: (n_frames, 2)    

        # Calculate velocity and acceleration with the correct FPS
        kp_velocity = self.calculate_velocity(kps, output_fps)
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
        c_eyes = self.normalize_features(feature=c_eyes, feature_type='c_eyes')
        c_lip = self.normalize_features(feature=c_lip, feature_type='c_lip')

        # Apply statistics-based normalization to derivatives
        kp_velocity = self.normalize_features(feature=kp_velocity, feature_type='kp_velocity')
        kp_acceleration = self.normalize_features(feature=kp_acceleration, feature_type='kp_acceleration')
        exp_velocity = self.normalize_features(feature=exp_velocity, feature_type='exp_velocity')
        exp_acceleration = self.normalize_features(feature=exp_acceleration, feature_type='exp_acceleration')
        
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
            'scale': scales,
            'c_eyes_lst': c_eyes,
            'c_lip_lst': c_lip,
            'metadata': metadata
        }

    def __len__(self):
        return len(self.pickle_paths)