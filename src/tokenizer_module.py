import torch.nn as nn
import torch
import yaml
import math
import pickle
import os
import shutil

from collections import namedtuple
from pathlib import Path
from huggingface_hub import ModelHubMixin, hf_hub_download, snapshot_download
from typing import Optional, Union, Dict

from utils.tokenizer_utils import load_fsq_vae, process_reconstruction
from src.utils.helper import clean_state_dict
from utils.tokenizer_utils import prepare_features
from src.motion_dataset import MotionDataset

FSQ_VALUES = namedtuple('fsq_values', ['L', 'D'])

class TokenizerModule(nn.Module, ModelHubMixin):
    def __init__(self, tokenizer_config: Union[str, Dict, None] = None):
        super().__init__()

        self.ds_path = Path("dataset")
        self.feat_dims = {
            'R': [1, 3, 3],
            'c_eyes_lst': [1, 2],
            'c_lip_lst': [1, 1],
            'exp': [1, 21, 3],
            'kp': [1, 21, 3],
            'scale': [1],
            't': [1, 3],
            'x_s': [1, 21, 3]
        }
        # Create a dataset instance for reusing utility methods
        self.dataset = MotionDataset(
            data_path=str(self.ds_path), 
            split='test',
            val_split=0.1,
            seed=2,
            compute_stats=False  # We'll load stats separately
        )
        self.mean = {k: v.to('cuda') for k, v in self.dataset.mean.items()}
        self.std = {k: v.to('cuda') for k, v in self.dataset.std.items()}

        # If tokenizer_config is None, we're likely loading from pretrained
        if tokenizer_config is not None:
            # Handle both string paths and dictionaries
            if isinstance(tokenizer_config, (str, Path)):
                self.tokenizer_config = yaml.safe_load(open(tokenizer_config, 'r'))['tokenizer_module']
            else:
                self.tokenizer_config = tokenizer_config

            # Load FSQ models with their specific configurations
            fsq_configs = {
                'lips': {'nfeats': 15},  # Lips model dimensions
                'exp': {'nfeats': 48},   # Expression model dimensions
                'rest': {'nfeats': 132},  # Rest features dimensions
                'rot_scale': {'nfeats': 10}  # Rotation and scale dimensions
            }

            # Load each FSQ model with its specific config
            self.lips_fsq, self.lips_feats = load_fsq_vae(
                Path(self.tokenizer_config['lips_path']),
                config_path=self.tokenizer_config.get('lips_path_config'),
                default_config=fsq_configs['lips']
            )
            self.exp_fsq, self.exp_feats = load_fsq_vae(
                Path(self.tokenizer_config['exp_path']),
                config_path=self.tokenizer_config.get('exp_path_config'),
                default_config=fsq_configs['exp']
            )
            self.rest_fsq, self.rest_feats = load_fsq_vae(
                Path(self.tokenizer_config['rest_path']),
                config_path=self.tokenizer_config.get('rest_path_config'),
                default_config=fsq_configs['rest']
            )
            self.rot_scale_fsq, self.rot_scale_feats = load_fsq_vae(
                Path(self.tokenizer_config['rot_scale_path']),
                config_path=self.tokenizer_config.get('rot_scale_path_config'),
                default_config=fsq_configs['rot_scale']
            )

            self.fsq_configs = {
                "lips": FSQ_VALUES(L=self.lips_fsq.fsq_levels, D=self.lips_fsq.fsq_dims),
                "exp": FSQ_VALUES(L=self.exp_fsq.fsq_levels, D=self.exp_fsq.fsq_dims),
                "rest": FSQ_VALUES(L=self.rest_fsq.fsq_levels, D=self.rest_fsq.fsq_dims),
                "rot_scale": FSQ_VALUES(L=self.rot_scale_fsq.fsq_levels, D=self.rot_scale_fsq.fsq_dims)
            }

            self.fsq_ranges = self._calculate_fsq_ranges(self.fsq_configs)
            
        # If tokenizer_config is None, the attributes will be loaded from the saved state
        self.exp_dims = {'exp': 48}
        self.lips_dims = {'exp': 15}
        self.rest_dims = {
            'c_eyes_lst': 2,
            'c_lip_lst': 1,
            'kp': 63,
            't': 3,
            'x_s': 63
        }
        self.rot_scale_dims = {'R': 9, 'scale': 1}

    def _save_pretrained(self, save_directory: str) -> None:
        """Save the model weights and its configuration file to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Create fsq_models directory and its config subdirectory
        fsq_models_dir = os.path.join(save_directory, "fsq_models")
        fsq_configs_dir = os.path.join(fsq_models_dir, "configs")
        os.makedirs(fsq_models_dir, exist_ok=True)
        os.makedirs(fsq_configs_dir, exist_ok=True)

        # Save the model weights
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        
        # Get the state dict
        state_dict = self.state_dict()
        
        # Save the model weights
        torch.save(state_dict, weights_path)
        
        # Save FSQ models and update config paths
        if hasattr(self, 'tokenizer_config'):
            config = self.tokenizer_config.copy()  # Make a copy to modify paths
            
            # Save each FSQ model and update its path in the config
            fsq_paths = ['lips_path', 'exp_path', 'rest_path', 'rot_scale_path']
            for path_key in fsq_paths:
                original_path = self.tokenizer_config[path_key]
                filename = os.path.basename(original_path)
                new_path = os.path.join(fsq_models_dir, filename)
                
                # Copy the FSQ model file
                shutil.copy2(original_path, new_path)
                
                # Try to copy the config file if it exists
                try:
                    config_src = Path(original_path).parent.parent / 'wandb' / 'latest-run' / 'files' / 'config.yaml'
                    if config_src.exists():
                        config_dst = os.path.join(fsq_configs_dir, f"{filename}.yaml")
                        shutil.copy2(config_src, config_dst)
                except Exception:
                    pass  # Skip if config doesn't exist
                
                # Update config with relative path
                config[path_key] = os.path.join("fsq_models", filename)
            
            # Save the updated config
            config_path = os.path.join(save_directory, "config.yaml")
            with open(config_path, 'w') as f:
                yaml.dump({'tokenizer_module': config}, f)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        **model_kwargs,
    ):
        """Load a model from a pretrained model on the Hugging Face Hub."""
        # Download the model weights and config
        try:
            # First try to download the config
            config_path = hf_hub_download(
                repo_id=model_id,
                filename="config.yaml",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
            
            # Then download the model weights
            weights_path = hf_hub_download(
                repo_id=model_id,
                filename="pytorch_model.bin",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
            
            # Load config if it exists
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)['tokenizer_module']
                    
                    # Download FSQ model files and their configs
                    fsq_files = ['lips_path', 'exp_path', 'rest_path', 'rot_scale_path']
                    for fsq_file in fsq_files:
                        original_path = config[fsq_file]
                        filename = os.path.basename(original_path)
                        
                        # Download the FSQ model file
                        downloaded_path = hf_hub_download(
                            repo_id=model_id,
                            filename=f"fsq_models/{filename}",
                            revision=revision,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            resume_download=resume_download,
                            token=token,
                            local_files_only=local_files_only,
                        )
                        
                        # Try to download the FSQ config file
                        try:
                            fsq_config = hf_hub_download(
                                repo_id=model_id,
                                filename=f"fsq_models/configs/{filename}.yaml",
                                revision=revision,
                                cache_dir=cache_dir,
                                force_download=force_download,
                                proxies=proxies,
                                resume_download=resume_download,
                                token=token,
                                local_files_only=local_files_only,
                            )
                        except Exception:
                            fsq_config = None
                        
                        # Update the config with the downloaded paths
                        config[fsq_file] = downloaded_path
                        if fsq_config:
                            config[f"{fsq_file}_config"] = fsq_config
                    
                    model_kwargs['tokenizer_config'] = config
            
            # Initialize model
            model = cls(**model_kwargs)
            
            # Load weights
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            
            return model
            
        except Exception as e:
            raise RuntimeError(
                f"Error loading model from HuggingFace Hub: {e}"
            ) from e

    def state_dict(self):
        """Get the state dict of the model including FSQ models and configurations."""
        state_dict = {}
        
        # Save FSQ models if they exist
        if hasattr(self, 'lips_fsq'):
            state_dict['lips_fsq'] = self.lips_fsq.state_dict()
            state_dict['exp_fsq'] = self.exp_fsq.state_dict()
            state_dict['rest_fsq'] = self.rest_fsq.state_dict()
            state_dict['rot_scale_fsq'] = self.rot_scale_fsq.state_dict()
            
            # Save FSQ configs
            state_dict['fsq_configs'] = {
                name: {'L': config.L, 'D': config.D}
                for name, config in self.fsq_configs.items()
            }
            
            # Save feature dimensions
            state_dict['feat_dims'] = self.feat_dims
            state_dict['exp_dims'] = self.exp_dims
            state_dict['lips_dims'] = self.lips_dims
            state_dict['rest_dims'] = self.rest_dims
            state_dict['rot_scale_dims'] = self.rot_scale_dims
            
            # Save dataset statistics
            state_dict['mean'] = {k: v.cpu() for k, v in self.mean.items()}
            state_dict['std'] = {k: v.cpu() for k, v in self.std.items()}
        
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a state dict into the model."""
        # Clean the state dict to remove any 'module.' prefixes
        state_dict = clean_state_dict(state_dict)
        
        # Load FSQ models
        if 'lips_fsq' in state_dict:
            self.lips_fsq.load_state_dict(state_dict['lips_fsq'])
            self.exp_fsq.load_state_dict(state_dict['exp_fsq'])
            self.rest_fsq.load_state_dict(state_dict['rest_fsq'])
            self.rot_scale_fsq.load_state_dict(state_dict['rot_scale_fsq'])
            
            # Load FSQ configs
            self.fsq_configs = {
                name: FSQ_VALUES(L=config['L'], D=config['D'])
                for name, config in state_dict['fsq_configs'].items()
            }
            
            # Recalculate FSQ ranges
            self.fsq_ranges = self._calculate_fsq_ranges(self.fsq_configs)
            
            # Load feature dimensions
            self.feat_dims = state_dict['feat_dims']
            self.exp_dims = state_dict['exp_dims']
            self.lips_dims = state_dict['lips_dims']
            self.rest_dims = state_dict['rest_dims']
            self.rot_scale_dims = state_dict['rot_scale_dims']
            
            # Load dataset statistics
            self.mean = {k: v.to('cuda') for k, v in state_dict['mean'].items()}
            self.std = {k: v.to('cuda') for k, v in state_dict['std'].items()}
            
        return super().load_state_dict({}, strict=False)

    def _calculate_fsq_ranges(self, fsq_configs):
        ranges = {}
        current_start = 0

        for fsq_name, (L, D) in fsq_configs.items():
            codebook_size = L ** D
            end_idx = current_start + codebook_size
            ranges[fsq_name] = (current_start, end_idx)
            current_start = end_idx

        return ranges
    

    def _prepare_feats(self, x: torch.Tensor):
        device = x.device
        rot_scale_feats = torch.cat([x[..., :9],   x[..., 138:139]], dim=-1).to(device) # (1, N, 10)
        rest_feats =      torch.cat([x[..., 9:12], x[..., 75:138], x[..., 139:205]], dim=-1).to(device) # (1, N, 132)
        exp_feats = x[..., 12:60].to(device) # (1, N, 48)
        lips_feats = x[..., 60:75].to(device) # (1, N, 15)

        return rest_feats, exp_feats, lips_feats, rot_scale_feats
    

    def _reconstruct_feats(self, feats: dict):
        self.exp_dims['exp'] = min(63, self.exp_dims['exp'] + self.lips_dims['exp']) # 48 + 15 = 63
        rest_reconstr = process_reconstruction(self.rest_dims, feats['rest'], False, self.std, self.mean)
        exp_reconstr = process_reconstruction(self.exp_dims, feats['exp'], False, self.std, self.mean)
        rot_scale_reconstr = process_reconstruction(self.rot_scale_dims, feats['rot_scale'], False, self.std, self.mean)

        return rest_reconstr, exp_reconstr, rot_scale_reconstr
    

    def _global_code_to_local_code(self, global_code: int, fsq_name: str):
        for _, (start_idx, end_idx) in self.fsq_ranges.items():
            if start_idx <= global_code < end_idx:
                local_code_index = global_code - start_idx
                return local_code_index
        
        raise ValueError(f"Global code {global_code} is out of range for any FSQ")


    def _local_code_to_global_code(self, local_code: int, fsq_name: str):        
        start_idx, end_idx = self.fsq_ranges[fsq_name]
        global_code = start_idx + local_code
        
        if global_code > end_idx:
            raise ValueError(f"Local code {local_code} is out of range for FSQ '{fsq_name}' (max local code: {end_idx - start_idx - 1})")
        
        return global_code
    

    def _prepare_features_dict(self, feature_dict: dict) -> torch.Tensor:
        metadata = feature_dict['metadata']
        feature_tensor = torch.zeros((1, metadata['n_frames'], 205), device="cuda")

        feature_tensor[..., :9] = feature_dict['R'].reshape(1, -1, 9)
        feature_tensor[..., 9:11] = feature_dict['c_eyes_lst'].reshape(1, -1, 2)
        feature_tensor[..., 11:12] = feature_dict['c_lip_lst'].reshape(1, -1, 1)
        feature_tensor[..., 12:75] = feature_dict['exp'].reshape(1, -1, 63)
        feature_tensor[..., 75:138] = feature_dict['kp'].reshape(1, -1, 63)
        feature_tensor[..., 138:139] = feature_dict['scale'].reshape(1, -1, 1)
        feature_tensor[..., 139:142] = feature_dict['t'].reshape(1, -1, 3)
        feature_tensor[..., 142:205] = feature_dict['x_s'].reshape(1, -1, 63)

        return feature_tensor


    def _dump_to_pickle(self, output: dict, pickle_path: str):
        pickle_dir = self.ds_path / "pickles"
        video_id = Path(pickle_path).stem

        new_path = pickle_dir / f"{video_id}_reconstructed.pkl"
        # new_path = "female_24fps_reconstructed.pkl"
        print(new_path)

        with open(new_path, "wb") as f:
            pickle.dump(output, f)
    

    @torch.no_grad()
    def features_to_codes(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        rest_feats, exp_feats, lips_feats, rot_scale_feats = self._prepare_feats(x)

        lips_indices = self.lips_fsq.encode(lips_feats)
        exp_indices = self.exp_fsq.encode(exp_feats)
        rest_indices = self.rest_fsq.encode(rest_feats)
        rot_scale_indices = self.rot_scale_fsq.encode(rot_scale_feats)

        lips_global_indices = torch.tensor([self._local_code_to_global_code(idx, 'lips') for idx in lips_indices[0]], device=device)
        exp_global_indices = torch.tensor([self._local_code_to_global_code(idx, 'exp') for idx in exp_indices[0]], device=device)
        rest_global_indices = torch.tensor([self._local_code_to_global_code(idx, 'rest') for idx in rest_indices[0]], device=device)
        rot_scale_global_indices = torch.tensor([self._local_code_to_global_code(idx, 'rot_scale') for idx in rot_scale_indices[0]], device=device)

        return torch.stack([lips_global_indices, exp_global_indices, rest_global_indices, rot_scale_global_indices], dim=0)


    @torch.no_grad()
    def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        device = codes.device
        lips_indices, exp_indices, rest_indices, rot_scale_indices = codes # (4, N_codes)

        lips_global_indices = torch.tensor([self._global_code_to_local_code(idx, 'lips') for idx in lips_indices], device=device)
        exp_global_indices = torch.tensor([self._global_code_to_local_code(idx, 'exp') for idx in exp_indices], device=device)
        rest_global_indices = torch.tensor([self._global_code_to_local_code(idx, 'rest') for idx in rest_indices], device=device)
        rot_scale_global_indices = torch.tensor([self._global_code_to_local_code(idx, 'rot_scale') for idx in rot_scale_indices], device=device)

        lips_reconstr = self.lips_fsq.decode(lips_global_indices.unsqueeze(0))
        exp_reconstr = self.exp_fsq.decode(exp_global_indices.unsqueeze(0))
        rest_reconstr = self.rest_fsq.decode(rest_global_indices.unsqueeze(0))
        rot_scale_reconstr = self.rot_scale_fsq.decode(rot_scale_global_indices.unsqueeze(0))

        exp_reconstr = torch.cat([exp_reconstr, lips_reconstr], dim=-1).to(device)

        reconstr_feats = {
            "exp": exp_reconstr,
            "rest": rest_reconstr,
            "rot_scale": rot_scale_reconstr
        }
        
        reconstr_rest, reconstr_exp, reconstr_rot_scale = self._reconstruct_feats(reconstr_feats)

        new_reconstr = torch.zeros((*reconstr_rest.shape[:-1], 205), device=device)
        new_reconstr[..., :9] = reconstr_rot_scale[..., :9]
        new_reconstr[..., 9:12] = reconstr_rest[..., :3]
        new_reconstr[..., 12:75] = reconstr_exp
        new_reconstr[..., 75:138] = reconstr_rest[..., 3:66]
        new_reconstr[..., 138:139] = reconstr_rot_scale[..., 9:10]
        new_reconstr[..., 139:142] = reconstr_rest[..., 66:69]
        new_reconstr[..., 142:205] = reconstr_rest[..., 69:132]

        return new_reconstr
    
    def _prepare_sample(self, sample: dict) -> torch.Tensor:
        rst_feats, self.rest_dims = prepare_features(sample, self.rest_feats, False, "cuda")
        exp_feats, self.exp_dims = prepare_features(sample, self.exp_feats, False, "cuda")
        lips_feats, self.lips_dims = prepare_features(sample, self.lips_feats, True, "cuda")
        rot_scale_feats, self.rot_scale_dims = prepare_features(sample, self.rot_scale_feats, False, "cuda")
        
        return rst_feats, exp_feats, lips_feats, rot_scale_feats

    def sample_to_features(self, sample: dict) -> torch.Tensor:
        feature_tensor = self._prepare_features_dict(sample)

        return feature_tensor

    

    def features_to_pickle(self, reconstr: torch.Tensor, pickle_path: str):
        # Get normalized tensors from reconstruction
        reconstr = reconstr.to('cpu').squeeze(0)
        frames = reconstr.shape[0]

        start = 0
        end = 0

        main_feats = {}

        # Unpack features from reconstruction tensor
        for feat, shape in self.feat_dims.items():
            if feat not in self.feat_dims:
                continue

            total_dim = math.prod(shape) # [1, 21, 3] -> 63

            end += total_dim
            # Shape (B, N, D)
            main_feats[feat] = reconstr[:, start:end].reshape(-1, *shape).numpy()
            
            start += total_dim
        
        # Pack features into output dictionary for pickle
        output = {
            "n_frames": frames,
            "output_fps": 24,
            "motion": [
                {
                    "kp": main_feats['kp'][i],
                    "exp": main_feats['exp'][i],
                    "x_s": main_feats['x_s'][i],
                    "t": main_feats['t'][i],
                    "R": main_feats['R'][i],
                    "scale": main_feats['scale'][i],
                } for i in range(frames)
            ],
            "c_eyes_lst": [main_feats['c_eyes_lst'][i] for i in range(frames)],
            "c_lip_lst": [main_feats['c_lip_lst'][i] for i in range(frames)],
        }

        self._dump_to_pickle(output=output, pickle_path=pickle_path)

        return output


    def pickle_to_features(self, pickle_path: str, device: str = 'cuda') -> dict:
        """
        Load and process a pickle file to extract features ready for tokenization.
        Uses the Dataset's process_pickle_path method to avoid code duplication.
        
        Args:
            pickle_path: Path to the pickle file
            device: Device to place the tensor on (default: 'cpu')
            
        Returns:
            Dictionary containing processed and normalized features
        """
        feature_dict = self.dataset.process_pickle_path(pickle_path)

        feature_tensor = self._prepare_features_dict(feature_dict)

        return feature_tensor
