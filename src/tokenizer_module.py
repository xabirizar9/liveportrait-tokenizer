from utils.tokenizer_utils import load_fsq_vae
import torch.nn as nn
import torch
from collections import namedtuple
from pathlib import Path
from utils.tokenizer_utils import load_fsq_vae, prepare_features, process_reconstruction
import yaml
import pickle
from huggingface_hub import PyTorchModelHubMixin

FSQ_VALUES = namedtuple('fsq_values', ['L', 'D'])

class TokenizerModule(nn.Module, PyTorchModelHubMixin):
    def __init__(self, tokenizer_config: str = None):
        super().__init__()
        
        # If tokenizer_config is None, we're likely loading from pretrained
        if tokenizer_config is not None:
            self.tokenizer_config = yaml.safe_load(open(tokenizer_config, 'r'))['tokenizer_module']

            self.lips_fsq, self.lips_feats = load_fsq_vae(Path(self.tokenizer_config['lips_path']))
            self.exp_fsq, self.exp_feats = load_fsq_vae(Path(self.tokenizer_config['exp_path']))
            self.rest_fsq, self.rest_feats = load_fsq_vae(Path(self.tokenizer_config['rest_path']))
            self.rot_scale_fsq, self.rot_scale_feats = load_fsq_vae(Path(self.tokenizer_config['rot_scale_path']))
            
            self.stats = pickle.load(open("dataset/stats_all.pkl", "rb"))

            # Send stats to GPU
            for key in self.stats['mean']:
                self.stats['mean'][key] = self.stats['mean'][key].to("cuda")
                self.stats['std'][key] = self.stats['std'][key].to("cuda")

            self.std = self.stats['std']
            self.mean = self.stats['mean']

            self.fsq_configs = {
                "lips": FSQ_VALUES(L=self.lips_fsq.fsq_levels, D=self.lips_fsq.fsq_dims),
                "exp": FSQ_VALUES(L=self.exp_fsq.fsq_levels, D=self.exp_fsq.fsq_dims),
                "rest": FSQ_VALUES(L=self.rest_fsq.fsq_levels, D=self.rest_fsq.fsq_dims),
                "rot_scale": FSQ_VALUES(L=self.rot_scale_fsq.fsq_levels, D=self.rot_scale_fsq.fsq_dims)
            }

            self.fsq_ranges = self._calculate_fsq_ranges(self.fsq_configs)
        # If tokenizer_config is None, the attributes will be loaded from the saved state

    def _calculate_fsq_ranges(self, fsq_configs):
        ranges = {}
        current_start = 0

        for fsq_name, (L, D) in fsq_configs.items():
            codebook_size = L ** D
            end_idx = current_start + codebook_size
            ranges[fsq_name] = (current_start, end_idx)
            current_start = end_idx

        return ranges
    
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

    def _prepare_feats(self, x):
        rest_feats, self.rest_dims = prepare_features(x, self.rest_feats, only_lips=False)
        exp_feats, self.exp_dims = prepare_features(x, self.exp_feats, only_lips=False)
        lips_feats, self.lips_dims = prepare_features(x, self.lips_feats, only_lips=True)
        rot_scale_feats, self.rot_scale_dims = prepare_features(x, self.rot_scale_feats, only_lips=False)

        return rest_feats, exp_feats, lips_feats, rot_scale_feats
    
    def _reconstruct_feats(self, feats: dict):
        self.exp_dims['exp'] = 63
        rest_reconstr = process_reconstruction(self.rest_dims, feats['rest'], False, self.std, self.mean)
        exp_reconstr = process_reconstruction(self.exp_dims, feats['exp'], False, self.std, self.mean)
        rot_scale_reconstr = process_reconstruction(self.rot_scale_dims, feats['rot_scale'], False, self.std, self.mean)

        return rest_reconstr, exp_reconstr, rot_scale_reconstr
    
    @torch.no_grad()
    def encode(self, x):
        rest_feats, exp_feats, lips_feats, rot_scale_feats = self._prepare_feats(x)

        lips_indices = self.lips_fsq.encode(lips_feats)
        exp_indices = self.exp_fsq.encode(exp_feats)
        rest_indices = self.rest_fsq.encode(rest_feats)
        rot_scale_indices = self.rot_scale_fsq.encode(rot_scale_feats)

        lips_global_indices = torch.tensor([self._local_code_to_global_code(idx, 'lips') for idx in lips_indices[0]])
        exp_global_indices = torch.tensor([self._local_code_to_global_code(idx, 'exp') for idx in exp_indices[0]])
        rest_global_indices = torch.tensor([self._local_code_to_global_code(idx, 'rest') for idx in rest_indices[0]])
        rot_scale_global_indices = torch.tensor([self._local_code_to_global_code(idx, 'rot_scale') for idx in rot_scale_indices[0]])

        return torch.stack([lips_global_indices, exp_global_indices, rest_global_indices, rot_scale_global_indices], dim=0)

    @torch.no_grad()
    def decode(self, codes: torch.Tensor):
        lips_indices = codes[0]
        exp_indices = codes[1]
        rest_indices = codes[2]
        rot_scale_indices = codes[3]

        lips_global_indices = torch.tensor([self._global_code_to_local_code(idx, 'lips') for idx in lips_indices], device='cuda')
        exp_global_indices = torch.tensor([self._global_code_to_local_code(idx, 'exp') for idx in exp_indices], device='cuda')
        rest_global_indices = torch.tensor([self._global_code_to_local_code(idx, 'rest') for idx in rest_indices], device='cuda')
        rot_scale_global_indices = torch.tensor([self._global_code_to_local_code(idx, 'rot_scale') for idx in rot_scale_indices], device='cuda')

        lips_reconstr = self.lips_fsq.decode(lips_global_indices.unsqueeze(0))
        exp_reconstr = self.exp_fsq.decode(exp_global_indices.unsqueeze(0))
        rest_reconstr = self.rest_fsq.decode(rest_global_indices.unsqueeze(0))
        rot_scale_reconstr = self.rot_scale_fsq.decode(rot_scale_global_indices.unsqueeze(0))

        exp_reconstr = torch.cat([exp_reconstr, lips_reconstr], dim=-1)

        reconstr_feats = {
            "exp": exp_reconstr,
            "rest": rest_reconstr,
            "rot_scale": rot_scale_reconstr
        }
        reconstr_rest, reconstr_exp, reconstr_rot_scale = self._reconstruct_feats(reconstr_feats)

        new_reconstr = torch.zeros((*reconstr_rest.shape[:-1], 205))
        new_reconstr[..., :9] = reconstr_rot_scale[..., :9]
        new_reconstr[..., 9:12] = reconstr_rest[..., :3]
        new_reconstr[..., 12:75] = reconstr_exp
        new_reconstr[..., 75:138] = reconstr_rest[..., 3:66]
        new_reconstr[..., 138:139] = reconstr_rest[..., 75:76]
        new_reconstr[..., 138:139] = reconstr_rot_scale[..., 9:10]
        new_reconstr[..., 139:142] = reconstr_rest[..., 66:69]
        new_reconstr[..., 142:205] = reconstr_rest[..., 69:132]

        return new_reconstr
