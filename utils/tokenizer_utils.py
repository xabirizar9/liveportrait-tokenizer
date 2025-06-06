from pathlib import Path
import torch
import yaml
from src.modules.vqvae import VQVae
from src.modules.fsq_vqvae import FSQVAE
from train_tokenizer import VQVAEModule
from torch.utils.data import Dataset

def load_fsq_vae(model_path: Path, config_path: Path = None, default_config: dict = None) -> VQVae:
    """
    Load and prepare a VQVAE model from a checkpoint file.
    
    Args:
        model_path (Path): Path to the model checkpoint file
        config_path (Path, optional): Path to the config file. If not provided, will try to find it in wandb directory
        default_config (dict, optional): Default configuration to use if no config file is found
        
    Returns:
        VQVae: Prepared VQVAE model loaded on CUDA and in eval mode
    """
    pretrained = torch.load(model_path)
    
    # If config_path is not provided, try to find it in wandb directory
    if config_path is None:
        if model_path.suffix == '.ckpt':
            config_path = model_path.parent.parent / 'wandb' / 'latest-run' / 'files' / 'config.yaml'
        else:
            config_path = model_path.parent / 'wandb' / 'latest-run' / 'files' / 'config.yaml'
    
    try:
        config = yaml.safe_load(open(config_path, "r"))
        feats_enabled = config['feats_enabled']['value']
        vqvae_config = config["vqvae"]['value']
    except (FileNotFoundError, KeyError):
        # If config not found or doesn't have the expected structure,
        # use default values that work for all FSQ models
        feats_enabled = {}
        
        # Base FSQ model configuration
        vqvae_config = {
            "output_emb_width": 64,
            "down_t": 2,
            "stride_t": 2,
            "width": 64,
            "depth": 3,
            "dilation_growth_rate": 3,
            "activation": "relu",
            "norm": None,
            "levels": [5, 5, 5, 5],  # FSQ specific
            "num_quantizers": 1,      # FSQ specific
            "use_quantization": True  # FSQ specific
        }
        
        # Update with model-specific config if provided
        if default_config is not None:
            vqvae_config.update(default_config)
    
    print([feat for feat in sorted(feats_enabled) if feats_enabled.get(feat, {}).get('enabled', False)])
    
    if model_path.suffix == '.ckpt':
        vqvae_module = VQVAEModule(vqvae_config=vqvae_config, losses_config={})
        vqvae_module.load_state_dict(pretrained['state_dict'])
        vqvae = vqvae_module.vqvae
    else:
        vqvae = FSQVAE(**vqvae_config)
        vqvae.load_state_dict(pretrained)

    vqvae.to("cuda")
    vqvae.eval()
    
    return vqvae, feats_enabled


def prepare_features(sample, feats_enabled, only_lips, device="cuda"):
    """
    Prepare features from a sample by processing enabled features and concatenating them.
    
    Args:
        sample (dict): Dictionary containing the sample data
        feats_enabled (dict): Dictionary of enabled features and their metadata
        device (str): Device to move tensors to (default: "cuda")
    
    Returns:
        tuple: (features tensor, dimensions dictionary)
    """
    frames = sample['kp'].shape[0]
    fps = sample['metadata']['output_fps']
    seq_len = min(sample['kp'].shape[0], 300)

    # Initialize an empty tensor list to collect features
    feature_tensors = []
    dims = {}

    for feat, metadata in feats_enabled.items():
        is_enabled = metadata['enabled']
        if is_enabled:
            print(f"Using {feat}")
            if feat in ["exp", "exp_velocity"]:
                if only_lips:
                    feature = sample[feat][:seq_len, :, 16:, :].reshape(1, seq_len, -1)
                else:
                    feature = sample[feat][:seq_len, :, :16, :].reshape(1, seq_len, -1)
            else:
                feature = sample[feat][:seq_len, ...].reshape(1, seq_len, -1)
            
            dims[feat] = feature.shape[-1]
            feature_tensors.append(feature)

    # Concatenate all enabled features
    if feature_tensors:
        features = torch.concat(feature_tensors, dim=2)
    else:
        # Create an empty tensor if no features are enabled
        features = torch.empty((1, seq_len, 0))

    features = features.to(device)
    print("dims: ", dims)
    print("Total dims: ", features.shape[-1])
    
    return features, dims


# Prepare output
def repackage_output(original: dict, reconstr: torch.Tensor, dataset: Dataset, dims: dict, feats_data: dict):
    # Get normalized tensors from reconstruction
    reconstr = reconstr.to('cpu').squeeze(0)
    frames = reconstr.shape[0]
    print("Frames: ", frames)

    start = 0
    end = 0

    main_feats = {
        "kp": None,
        "exp": None,
        "x_s": None,
        "t": None,
        "R": None,
        "scale": None,
        "c_eyes_lst": None,
        "c_lip_lst": None
    }

    for feat, metadata in feats_data.items():
        is_enabled = metadata['enabled']
        feat_shape = metadata['shape']

        if is_enabled and feat in main_feats:
            end += dims[feat]
            print(f"{feat}: {start}:{end}")
            if feat == "exp":
                rec_feat = reconstr[:, start:end].reshape(-1, *feat_shape)
            else:
                rec_feat = reconstr[:, start:end].reshape(-1, *feat_shape)
            main_feats[feat] = rec_feat
            start += dims[feat]
    
    # Denormalize original features too (since they come from __getitem__)
    orig_kps = dataset.denormalize_features(original['kp'], "kp")
    orig_exp = dataset.denormalize_features(original['exp'], "exp")
    orig_x_s = dataset.denormalize_features(original['x_s'], "x_s")
    orig_t = dataset.denormalize_features(original['t'], "t")
    orig_R = dataset.denormalize_features(original['R'], "R")
    orig_scale = dataset.denormalize_features(original['scale'], "scale").squeeze(-1)
    orig_c_eyes_lst = dataset.denormalize_features(original['c_eyes_lst'], "c_eyes_lst")
    orig_c_lip_lst = dataset.denormalize_features(original['c_lip_lst'], "c_lip_lst")

    # Print which features are enabled and will be used
    print("Enabled features:")
    for feat, metadata in feats_data.items():
        if metadata['enabled'] and 'velocity' not in feat and 'acceleration' not in feat:
            print(f"- {feat}")

    n_frames = min(original['metadata']['n_frames'], frames)

    output = {
        "n_frames": n_frames,
        "output_fps": original['metadata']['output_fps'],
        "motion": [
            {
                "kp": main_feats['kp'][i].cpu().numpy() if feats_data['kp']['enabled'] else orig_kps[i].cpu().numpy(),
                "exp": main_feats['exp'][i].cpu().numpy() if feats_data['exp']['enabled'] else orig_exp[i].cpu().numpy(),
                "x_s": main_feats['x_s'][i].cpu().numpy() if feats_data['x_s']['enabled'] else orig_x_s[i].cpu().numpy(),
                "t": main_feats['t'][i].cpu().numpy() if feats_data['t']['enabled'] else orig_t[i].cpu().numpy(),
                "R": main_feats['R'][i].cpu().numpy() if feats_data['R']['enabled'] else orig_R[i].cpu().numpy(),
                "scale": main_feats['scale'][i].cpu().numpy() if feats_data['scale']['enabled'] else orig_scale[i].cpu().numpy(),
            } for i in range(frames)
        ],
        "c_eyes_lst": [main_feats['c_eyes_lst'][i].cpu().numpy() if feats_data['c_eyes_lst']['enabled'] else orig_c_eyes_lst[i].cpu().numpy() for i in range(frames)],
        "c_lip_lst": [main_feats['c_lip_lst'][i].cpu().numpy() if feats_data['c_lip_lst']['enabled'] else orig_c_lip_lst[i].cpu().numpy() for i in range(frames)],
    }
    return output


def process_reconstruction(dims, reconstr, exp_lips, std, mean):
    """
    Process reconstruction tensor by filtering out velocity dimensions and applying normalization.
    
    Args:
        dims (dict): Dictionary containing feature dimensions
        reconstr (torch.Tensor): Reconstruction tensor
        std (dict): Dictionary of standard deviations for normalization
        mean (dict): Dictionary of means for normalization
        
    Returns:
        torch.Tensor: Processed reconstruction tensor
    """
    # Filter out velocity dimensions and add lip dimensions to exp
    filtered_dims = {k: v for k, v in dims.items() if k not in ['kp_velocity', 'kp_acceleration', 'exp_velocity']}

    total_dims = sum(filtered_dims.values())

    # Initialize output tensor
    new_reconstr = torch.zeros((*reconstr.shape[:-1], total_dims))

    cur_ind = 0
    reconstr_ind = 0
    for feat, indices in dims.items():
        if 'velocity' in feat:
            # Skip velocity in new_reconstr but still increment reconstr_ind
            reconstr_ind += indices
            continue
            
        print(f"{feat} {cur_ind} : {cur_ind + indices}")
        new_reconstr[..., cur_ind:cur_ind + indices] = reconstr[..., reconstr_ind:reconstr_ind + indices] * std[feat] + mean[feat]
            
        cur_ind += indices
        reconstr_ind += indices

    return new_reconstr


def merge_tokens(audio_tokens, motion_tokens):
    """
    Merge audio and motion tokens by concatenating motion tokens after SNAC tokens.
    
    Args:
        audio_tokens: List of 3 tensors from SNAC encoder [level0, level1, level2]
        motion_tokens: Tensor of shape (4, N) containing motion tokens
    
    Returns:
        Tensor of shape [num_frames, 15] with SNAC tokens followed by motion tokens
    """
    frame_tokens = []

    for i in range(audio_tokens[0].shape[-1]):
        # Extract tokens for frame i
        snac_1 = audio_tokens[0][:, i      ] + 128266
        snac_2 = audio_tokens[1][:, 2*i    ] + 128266 +   4096
        snac_3 = audio_tokens[2][:, 4*i    ] + 128266 + 2*4096
        snac_4 = audio_tokens[2][:, 4*i + 1] + 128266 + 3*4096
        snac_5 = audio_tokens[1][:, 2*i + 1] + 128266 + 4*4096
        snac_6 = audio_tokens[2][:, 4*i + 2] + 128266 + 5*4096
        snac_7 = audio_tokens[2][:, 4*i + 3] + 128266 + 6*4096

        audio_frame_tokens = torch.cat([snac_1, snac_2, snac_3, snac_4, snac_5, snac_6, snac_7])

        mot_tokens = motion_tokens[:, i*2:(i+1)*2].flatten() # (4, 2) -> (8,)
        mot_tokens += 128266 + 7*4096
        
        # Concatenate audio tokens followed by motion tokens
        frame_token = torch.cat([audio_frame_tokens, mot_tokens])
        frame_tokens.append(frame_token)

    # Convert to tensor: [num_frames, 15] (7 audio + 8 motion)
    return torch.stack(frame_tokens)


def unmerge_tokens(frame_tokens):
    """
    Unmerge concatenated frame tokens back into separate audio and motion tokens.
    
    Args:
        frame_tokens: Tensor of shape [num_frames, 15] with audio tokens followed by motion tokens
    
    Returns:
        Tuple of (audio_tokens, motion_tokens) where:
        - audio_tokens: List of 3 tensors [level0, level1, level2] 
        - motion_tokens: Tensor of shape (4, N)
    """
    num_frames = frame_tokens.shape[0]
    
    # Initialize output tensors
    audio_level0 = torch.zeros((1, num_frames), dtype=torch.long, device=frame_tokens.device)
    audio_level1 = torch.zeros((1, num_frames * 2), dtype=torch.long, device=frame_tokens.device)
    audio_level2 = torch.zeros((1, num_frames * 4), dtype=torch.long, device=frame_tokens.device)
    motion_tokens = torch.zeros((4, num_frames * 2), dtype=torch.long, device=frame_tokens.device)
    
    for i in range(num_frames):
        frame_token = frame_tokens[i]
        
        # Extract audio tokens (first 7 tokens)
        audio_frame_tokens = frame_token[:7]
        
        # Extract motion tokens (last 8 tokens)
        mot_tokens = frame_token[7:] - (128266 + 7*4096)  # Remove offset
        motion_tokens[:, i*2:(i+1)*2] = mot_tokens.reshape(4, 2)
        
        # Reconstruct SNAC tokens by removing offsets
        snac_1 = audio_frame_tokens[0] - 128266
        snac_2 = audio_frame_tokens[1] - (128266 + 4096)
        snac_3 = audio_frame_tokens[2] - (128266 + 2*4096)
        snac_4 = audio_frame_tokens[3] - (128266 + 3*4096)
        snac_5 = audio_frame_tokens[4] - (128266 + 4*4096)
        snac_6 = audio_frame_tokens[5] - (128266 + 5*4096)
        snac_7 = audio_frame_tokens[6] - (128266 + 6*4096)
        
        # Place tokens back into their respective levels
        audio_level0[:, i] = snac_1
        audio_level1[:, 2*i] = snac_2
        audio_level1[:, 2*i + 1] = snac_5
        audio_level2[:, 4*i] = snac_3
        audio_level2[:, 4*i + 1] = snac_4
        audio_level2[:, 4*i + 2] = snac_6
        audio_level2[:, 4*i + 3] = snac_7
    
    return [audio_level0, audio_level1, audio_level2], motion_tokens