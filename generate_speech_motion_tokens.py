import torch
import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

import torchaudio
from snac import SNAC
from transformers import AutoTokenizer

from src import TokenizerModule
from src.full_dataset import SNACMotionTextDataset
from src.utils.rprint import rlog as log


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def create_input_ids(speech_motion_tokens, text_tokens):
    """
    Create input_ids following the format from the notebook.
    
    Args:
        speech_motion_tokens: Tensor of shape [num_frames, 15]
        text_tokens: List of text token IDs
    
    Returns:
        Dictionary with input_ids, labels, and attention_mask
    """
    # Token definitions
    tokeniser_length = 128256
    start_of_text = 128000
    end_of_text = 128009
    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2
    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4
    start_of_ai = tokeniser_length + 5
    end_of_ai = tokeniser_length + 6
    pad_token = tokeniser_length + 7

    # Flatten speech motion tokens
    speech_motion_tokens_flat = speech_motion_tokens.flatten().tolist()
    
    # Add end of text token to text tokens
    text_tokens_copy = text_tokens.copy()
    text_tokens_copy.append(end_of_text)

    input_ids = (
        [start_of_human]
        + text_tokens_copy
        + [end_of_human]
        + [start_of_ai]
        + [start_of_speech]
        + speech_motion_tokens_flat
        + [end_of_speech]
        + [end_of_ai]
    )

    output = {
        "input_ids": input_ids,
        "labels": input_ids,
        "attention_mask": [1] * len(input_ids)
    }
    
    return output


class FilteredDataset(SNACMotionTextDataset):
    """Wrapper around the base dataset to filter out already processed videos."""
    def __init__(self, base_dataset, output_dir):
        # Copy all attributes from base dataset
        self.__dict__.update(base_dataset.__dict__)
        self.base_dataset = base_dataset
        self.output_dir = Path(output_dir)
        
        # Create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter out video paths that already have .pt files
        self.valid_indices = []
        for i in range(len(base_dataset)):
            pickle_path = base_dataset.pickle_paths[i]
            pt_path = self.output_dir / f"{pickle_path.stem}.pt"
            if not pt_path.exists():
                self.valid_indices.append(i)
        
        # Log the filtering results
        skipped = len(base_dataset) - len(self.valid_indices)
        log(f"Filtering: {skipped} videos already processed, {len(self.valid_indices)} videos to process")

    def __getitem__(self, idx):
        # Get the actual index from our filtered list
        orig_idx = self.valid_indices[idx]
        return self.base_dataset[orig_idx]

    def __len__(self):
        return len(self.valid_indices)


class TokenGenerationModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False  # No optimization needed
        
        # Store config for later use
        self.config = config
        
        # Initialize with default device ID, will be updated in setup
        self.device_id = 0
        
        # Create output directory for tokens
        self.output_dir = Path(config['output_path'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Will initialize tokenizers in setup to ensure correct device assignment
        self.audio_tokenizer = None
        self.motion_tokenizer = None
        self.text_tokenizer = None

    def setup(self, stage=None):
        """Setup method called after trainer is initialized but before training/testing."""
        # Now we can safely access self.trainer
        self.device_id = self.trainer.local_rank if self.trainer is not None else 0
        
        # Initialize tokenizers
        self.init_tokenizers()

    def init_tokenizers(self):
        """Initialize tokenizers on the correct device."""
        if self.audio_tokenizer is None:
            log(f"Loading tokenizers on GPU {self.device_id}")
            
            # Load audio tokenizer (SNAC)
            self.audio_tokenizer = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
            
            # Load motion tokenizer
            self.motion_tokenizer = TokenizerModule.from_pretrained("InternalCan/tokenizer_module")
            
            # Load text tokenizer
            self.text_tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")
            
            log(f"Tokenizers loaded successfully on GPU {self.device_id}")

    def test_step(self, batch, batch_idx):
        """Process a single sample batch."""
        sample = batch
        
        # Extract sample information - motion contains the metadata
        pickle_path = Path(sample['motion']['metadata']['pickle_path'])
        filename = pickle_path.stem
        output_path = self.output_dir / f"{filename}.pt"
        
        # Skip if output .pt file already exists (double check)
        if output_path.exists():
            log(f"Skipping {filename} - .pt file already exists at {output_path}")
            return None
        
        try:
            # Extract components
            audio = sample['audio']
            motion = sample['motion']
            text = sample['text']
            
            # Resample motion to 24 FPS to match audio
            motion_resampled = self.base_dataset.resample_item(motion, 24)
            
            # Generate tokens
            with torch.no_grad():
                # Motion tokens
                motion_features = self.motion_tokenizer.sample_to_features(motion_resampled)
                motion_tokens = self.motion_tokenizer.features_to_codes(motion_features)
                
                # Audio tokens
                audio_tokens = self.audio_tokenizer.encode(audio.unsqueeze(0).to('cuda'))
                
                # Text tokens
                text_tokens = self.text_tokenizer.encode(text, add_special_tokens=True)
                
                # Merge audio and motion tokens
                speech_motion_tokens = merge_tokens(audio_tokens, motion_tokens)
                
                # Create final input format
                token_data = create_input_ids(speech_motion_tokens, text_tokens)
                
                # Add metadata
                token_data['metadata'] = {
                    'pickle_path': str(pickle_path),
                    'text': text,
                    'n_frames': motion_resampled['metadata']['n_frames'],
                    'output_fps': motion_resampled['metadata']['output_fps']
                }
                
                # Save to .pt file
                torch.save(token_data, output_path)
                log(f"Saved tokens to {output_path}")
                
        except Exception as e:
            log(f"Error processing {filename}: {str(e)}")
            # Log the failed sample to a file
            faulty_samples_path = self.output_dir / "faulty_samples.txt"
            with open(faulty_samples_path, "a") as f:
                f.write(f"{filename}: {str(e)}\n")
            return None

    def configure_optimizers(self):
        """No optimizer needed for token generation."""
        return None


def custom_collate_fn(batch):
    """Custom collate function to handle single samples without stacking."""
    # Since we're processing one sample at a time, just return the first item
    return batch[0]


def main(config):
    # Set up base dataset
    base_dataset = SNACMotionTextDataset(
        config['data_path'], 
        split=config.get('split', 'train'), 
        val_split=config.get('val_split', 0.2),
        seed=config.get('seed', 42),
        compute_stats=False,
        device='cuda'
    )
    
    # Filter out already processed samples
    dataset = FilteredDataset(base_dataset, config['output_path'])
    
    if len(dataset) == 0:
        print("All samples have already been processed. Nothing to do.")
        return
        
    print(f"Found {len(dataset)} samples to process")

    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time
        num_workers=config.get('num_workers', 0),
        shuffle=False,
        persistent_workers=True if config.get('num_workers', 0) > 0 else False,
        collate_fn=custom_collate_fn
    )

    # Initialize model
    model = TokenGenerationModule(config)
    
    # Store reference to base dataset for resampling
    model.base_dataset = base_dataset

    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",  # Use all available GPUs
        strategy="ddp",  # Use distributed data parallel
        precision="32",
        default_root_dir=str(config['output_path']),
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    # Run token generation
    trainer.test(model, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/token_generation.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Convert string paths to Path objects
    config['data_path'] = Path(config['data_path'])
    config['output_path'] = Path(config['output_path'])

    main(config)


