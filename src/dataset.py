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
        all_features = []

        print(f"Computing statistics for {len(pickle_paths)} samples")
        
        for pickle_path in pickle_paths:    
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            motion = data['motion']
            
            # Extract keypoints and expressions
            for m in motion:
                # Stack keypoints and expressions horizontally
                kp = torch.tensor(m['kp']).reshape(-1, 63)  # Reshape to (frames, 21*3)
                exp = torch.tensor(m['exp']).reshape(-1, 63)  # Reshape to (frames, 21*3)
                x_s = torch.tensor(m['x_s']).reshape(-1, 63)  # Reshape to (frames, 21*3)
                t = torch.tensor(m['t']).reshape(-1, 3)  # Reshape to (frames, 3)
                R = torch.tensor(m['R']).reshape(-1, 9)  # Reshape to (frames, 3, 3)
                scale = torch.tensor(m['scale']).reshape(-1, 1)  # Reshape to (frames,)
                features = torch.cat([kp, exp, x_s, t, R, scale], dim=1)  # Shape: (frames, 201)
                all_features.append(features)
        
        # Stack all features (shape: total_frames, feature_dim)
        all_features = torch.cat(all_features, dim=0)
        
        # Compute statistics along the first dimension (per feature)
        self.mean = all_features.mean(dim=0)  # Shape: (feature_dim,)
        self.std = all_features.std(dim=0)  # Shape: (feature_dim,)
        
        # Replace any zero std with 1 to avoid division by zero
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        
        # Save statistics
        with open(stats_path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)
        
        print(f"Computed feature-wise statistics: mean shape={self.mean.shape}, std shape={self.std.shape}")

    def __getitem__(self, idx):
        pickle_path = self.pickle_paths[idx]
        
        # Load pickle file
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract required data
        n_frames = data['n_frames']
        output_fps = data['output_fps']
        motion = data['motion']
        c_eyes_lst = data['c_eyes_lst']
        c_lip_lst = data['c_lip_lst']
        

        # Extract keypoints for all frames
        kps = []
        exps = []
        x_s = []
        translations = []
        rotations = []
        scales = []
        c_eyes = []
        c_lip = []

        for i, m in enumerate(motion):
            kps.append(torch.tensor(m['kp']))  # Shape: (n_frames, 1, 21, 3)
            exps.append(torch.tensor(m['exp']))  # Shape: (n_frames, 1, 21, 3)
            x_s.append(torch.tensor(m['x_s']))  # Shape: (n_frames, 1, 21, 3)
            translations.append(torch.tensor(m['t']))  # Shape: (n_frames, 1, 3)
            rotations.append(torch.tensor(m['R']))  # Shape: (n_frames, 3, 3)
            scales.append(torch.tensor(m['scale']))  # Shape: (n_frames,)
            c_eyes.append(torch.tensor(c_eyes_lst[i]).squeeze(0))  # Shape: (n_frames, 2)
            c_lip.append(torch.tensor(c_lip_lst[i]).squeeze(0))  # Shape: (n_frames, 2)

        kps = torch.stack(kps)  # Shape: (n_frames, 1, 21, 3)
        exps = torch.stack(exps)  # Shape: (n_frames, 1, 21, 3)
        x_s = torch.stack(x_s)  # Shape: (n_frames, 1, 21, 3)
        translations = torch.stack(translations)  # Shape: (n_frames, 1, 3)
        rotations = torch.stack(rotations)  # Shape: (n_frames, 3, 3)
        scales = torch.stack(scales)  # Shape: (n_frames,)

        # Stack c_eyes and c_lip lists
        c_eyes_lst = torch.stack(c_eyes)  # Shape: (n_frames, 2)
        c_lip_lst = torch.stack(c_lip)  # Shape: (n_frames, 2)
    
        # Reshape tensors for proper broadcasting
        kp_shape = kps.shape
        exp_shape = exps.shape
        x_s_shape = x_s.shape
        t_shape = translations.shape
        R_shape = rotations.shape
        scale_shape = scales.shape
        
        # Get the mean and std for keypoints (first 63 features)
        kp_mean = self.mean[:63]
        kp_std = self.std[:63]
        
        # Get the mean and std for expressions (next 63 features)
        exp_mean = self.mean[63:126]
        exp_std = self.std[63:126]
        
        # Get the mean and std for x_s (next 63 features)
        x_s_mean = self.mean[126:189]
        x_s_std = self.std[126:189]
        
        # Get the mean and std for translations (next 3 features)
        t_mean = self.mean[189:192]
        t_std = self.std[189:192]
        
        # Get the mean and std for rotations (next 9 features)
        R_mean = self.mean[192:201]
        R_std = self.std[192:201]
        
        # Get the mean and std for scales (next 1 feature)
        scale_mean = self.mean[201]
        scale_std = self.std[201]
        
        # Reshape kps and apply normalization
        kps = kps.reshape(kp_shape[0], -1)  # Flatten to (n_frames, 63)
        kps = (kps - kp_mean) / kp_std
        kps = kps.reshape(kp_shape)  # Reshape back to original shape
        
        # Reshape exps and apply normalization
        exps = exps.reshape(exp_shape[0], -1)  # Flatten to (n_frames, 63)
        exps = (exps - exp_mean) / exp_std
        exps = exps.reshape(exp_shape)  # Reshape back to original shape

        # Reshape x_s and apply normalization
        x_s = x_s.reshape(x_s_shape[0], -1)  # Flatten to (n_frames, 63)
        x_s = (x_s - x_s_mean) / x_s_std
        x_s = x_s.reshape(x_s_shape)  # Reshape back to original shape
        
        # Reshape translations and apply normalization
        translations = translations.reshape(t_shape[0], -1)  # Flatten to (n_frames, 3)
        translations = (translations - t_mean) / t_std
        translations = translations.reshape(t_shape)  # Reshape back to original shape
        
        # Reshape rotations and apply normalization
        rotations = rotations.reshape(R_shape[0], -1)  # Flatten to (n_frames, 9)
        rotations = (rotations - R_mean) / R_std
        rotations = rotations.reshape(R_shape)  # Reshape back to original shape
        
        # Apply normalization to scales
        scales = (scales - scale_mean) / scale_std
        # Metadata
        metadata = {
            'pickle_path': str(pickle_path),
            'n_frames': n_frames,
            'output_fps': output_fps
        }
        
        # Return the structured data
        return {
            'kp': kps,
            'exp': exps,
            'x_s': x_s,
            't': translations,
            'R': rotations,
            'scale': scales,
            'c_eyes_lst': c_eyes_lst,
            'c_lip_lst': c_lip_lst,
            'metadata': metadata
        }

    def __len__(self):
        return len(self.pickle_paths)