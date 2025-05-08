import torch
import numpy as np
import pandas as pd
import cv2
import imageio
import os
import os.path as osp
import yaml
from pathlib import Path
from argparse import ArgumentParser
import sys
from tqdm import tqdm
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from src.modules.motion_extractor import MotionExtractor
from src.live_portrait_pipeline import LivePortraitPipeline
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.video import get_fps
from src.utils.io import dump
from src.utils.helper import is_square_video
from train_tokenizer import Dataset
from src.utils.rprint import rlog as log


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def partial_fields(target_class, kwargs):
    """Extract fields from kwargs that match attributes in target_class."""
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


class FilteredDataset(torch.utils.data.Dataset):
    """Wrapper around the base dataset to filter out already processed videos."""
    def __init__(self, base_dataset, output_dir):
        self.base_dataset = base_dataset
        self.output_dir = Path(output_dir)
        
        # Create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter out video paths that already have pickles
        self.valid_indices = []
        for i in range(len(base_dataset)):
            video_path = base_dataset.video_paths[i]
            pickle_path = self.output_dir / f"{video_path.stem}.pkl"
            if not pickle_path.exists():
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


class BatchedPreprocessingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False  # No optimization needed
        
        # Store config for later use
        self.config = config
        
        # Initialize with default device ID, will be updated in setup
        self.device_id = 0
        
        # Create output directory for templates
        self.template_dir = Path(config['output_path'])
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Will initialize pipeline in setup to ensure correct device assignment
        self.pipeline = None
        self.live_portrait_wrapper = None
        self.cropper = None
        
        # Batch size for frame processing
        self.frame_batch_size = config.get('frame_batch_size', 32)
        
        # Deferred initialization of inference config
        self.inference_cfg = None
        self.crop_cfg = None

    def setup(self, stage=None):
        """Setup method called after trainer is initialized but before training/testing."""
        # Now we can safely access self.trainer
        self.device_id = self.trainer.local_rank if self.trainer is not None else 0
        
        # Initialize inference config with specific device ID
        inference_cfg_dict = dict(self.config)
        inference_cfg_dict['device_id'] = self.device_id
        
        # Initialize inference and crop configs
        self.inference_cfg = partial_fields(InferenceConfig, inference_cfg_dict)
        self.crop_cfg = partial_fields(CropConfig, inference_cfg_dict)
        
        # Set flag force CPU to False to ensure GPU usage
        self.inference_cfg.flag_force_cpu = False
        
        # Initialize the pipeline
        self.init_pipeline()

    def init_pipeline(self):
        """Initialize pipeline on the correct device."""
        if self.pipeline is None:
            # Force the device_id to be the current GPU rank
            self.inference_cfg.device_id = self.device_id
            
            # Initialize the LivePortraitPipeline
            self.pipeline = LivePortraitPipeline(
                inference_cfg=self.inference_cfg,
                crop_cfg=self.crop_cfg
            )
            
            # Access the wrapper and cropper from the pipeline
            self.live_portrait_wrapper = self.pipeline.live_portrait_wrapper
            self.cropper = self.pipeline.cropper
            
            # Log device information
            log(f"Process running on GPU {self.device_id} with device: {self.live_portrait_wrapper.device}")

    def test_step(self, batch, batch_idx):
        """Process a single video batch."""
        video_tensor, metadata = batch
        video_path = metadata['video_path'][0]  # Get first item since batch size is 1
        output_fps = float(metadata['fps'][0])
        

        # Skip if output pickle already exists (double check)
        filename = str(Path(video_path).stem)
        template_path = self.template_dir / f"{filename}.pkl"
        if template_path.exists():
            log(f"Skipping {video_path} - pickle already exists at {template_path}")
            return None
            
        # Extract frames from the preprocessed tensor
        # Handle the batch dimension properly
        if video_tensor.dim() == 6:  # [B, T, 1, 3, H, W]
            frames = video_tensor.squeeze(0).squeeze(1)  # Remove batch and channel dims
        else:  # [T, 1, 3, H, W]
            frames = video_tensor.squeeze(1)  # Remove channel dim
        
        # Convert to numpy and scale
        frames = frames.permute(0, 2, 3, 1).cpu().numpy() * 255.0
        frames = frames.astype(np.uint8)
        
        log(f"Using original FPS: {output_fps} for {video_path}")

        # Process driving video
        log(f"Processing video: {video_path}")
        driving_rgb_lst = frames
        driving_n_frames = len(driving_rgb_lst)
        
        driving_lmk_crop_lst = None
        driving_rgb_crop_256x256_lst = None
        
        log("Using video without cropping...")
        driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
        driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]
        c_d_eyes_lst, c_d_lip_lst = self.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
        
        # Prepare videos for template creation
        I_d_lst = self.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
        
        # Create motion template
        driving_template_dct = self.pipeline.make_motion_template(
            I_d_lst, 
            c_d_eyes_lst, 
            c_d_lip_lst, 
            output_fps=output_fps
        )
        
        # Save template
        with open(template_path, 'wb') as f:
            pickle.dump(driving_template_dct, f)
        log(f"Saved motion template to {template_path}")

    def configure_optimizers(self):
        """No optimizer needed for preprocessing."""
        return None


def custom_collate_fn(batch):
    """Custom collate function to handle the metadata dictionary format."""
    video_tensors = [item[0] for item in batch]
    metadata = {
        'video_path': [item[1]['video_path'] for item in batch],
        'fps': [item[1]['fps'] for item in batch]
    }
    
    # Stack video tensors
    video_tensors = torch.stack(video_tensors)
    
    return video_tensors, metadata


def main(config):
    # Set up base dataset
    base_dataset = Dataset(config['data_path'], split='train', val_split=0.0)
    
    # Filter out already processed videos
    dataset = FilteredDataset(base_dataset, config['output_path'])
    
    if len(dataset) == 0:
        print("All videos have already been processed. Nothing to do.")
        return
        
    print(f"Found {len(dataset)} videos to process")

    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one video at a time
        num_workers=config['num_workers'],
        shuffle=False,
        persistent_workers=True if config['num_workers'] > 0 else False,
        collate_fn=custom_collate_fn
    )

    # Initialize model
    model = BatchedPreprocessingModule(config)

    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="32",
        default_root_dir=str(config['output_path']),
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Run preprocessing
    trainer.test(model, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/preprocess.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Add frame batch size if not present
    if 'frame_batch_size' not in config:
        config['frame_batch_size'] = 32
    
    # Convert string paths to Path objects
    config['data_path'] = Path(config['data_path'])
    config['output_path'] = Path(config['output_path'])

    main(config) 