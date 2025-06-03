import torch.nn as nn
import torch
import yaml
import math
import pickle

from collections import namedtuple
from pathlib import Path
from huggingface_hub import PyTorchModelHubMixin

from utils.tokenizer_utils import load_fsq_vae, process_reconstruction

from src.dataset import Dataset

FSQ_VALUES = namedtuple('fsq_values', ['L', 'D'])

class TokenizerModule(nn.Module, PyTorchModelHubMixin):
    def __init__(self, tokenizer_config: str = None):
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
        self.dataset = Dataset(
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
            self.tokenizer_config = yaml.safe_load(open(tokenizer_config, 'r'))['tokenizer_module']

            self.lips_fsq, self.lips_feats = load_fsq_vae(Path(self.tokenizer_config['lips_path']))
            self.exp_fsq, self.exp_feats = load_fsq_vae(Path(self.tokenizer_config['exp_path']))
            self.rest_fsq, self.rest_feats = load_fsq_vae(Path(self.tokenizer_config['rest_path']))
            self.rot_scale_fsq, self.rot_scale_feats = load_fsq_vae(Path(self.tokenizer_config['rot_scale_path']))

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


    def _dump_to_pickle(self, output: dict, pickle_path: str):
        pickle_dir = self.ds_path / "pickles"
        video_id = Path(pickle_path).stem

        new_path = pickle_dir / f"{video_id}_reconstructed.pkl"
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
    

    def features_to_pickle(self, original: dict, reconstr: torch.Tensor, pickle_path: str):
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
            "output_fps": original['metadata']['output_fps'],
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
        metadata = feature_dict['metadata']

        feature_tensor = torch.zeros((1, metadata['n_frames'], 205), device=device)

        feature_tensor[..., :9] = feature_dict['R'].reshape(1, -1, 9)
        feature_tensor[..., 9:11] = feature_dict['c_eyes_lst'].reshape(1, -1, 2)
        feature_tensor[..., 11:12] = feature_dict['c_lip_lst'].reshape(1, -1, 1)
        feature_tensor[..., 12:75] = feature_dict['exp'].reshape(1, -1, 63)
        feature_tensor[..., 75:138] = feature_dict['kp'].reshape(1, -1, 63)
        feature_tensor[..., 138:139] = feature_dict['scale'].reshape(1, -1, 1)
        feature_tensor[..., 139:142] = feature_dict['t'].reshape(1, -1, 3)
        feature_tensor[..., 142:205] = feature_dict['x_s'].reshape(1, -1, 63)

        return feature_tensor
        