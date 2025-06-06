import torch
import numpy as np
import pandas as pd
import os
import pytorch_lightning as pl
import wandb
import yaml
import signal
import sys

import torch.nn.functional as F
import torch.distributed
import torch.nn as nn

from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from pprint import pprint

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


from src.modules.vqvae import VQVae
from src.modules.fc_vqvae import FCVQVae
from src.modules.res_vqvae import ResVQVae
from src.motion_dataset import MotionDataset
from src.data_collator import collate_fn


class VQVAEModule(pl.LightningModule):
    def __init__(self, vqvae_config, losses_config, lr=1e-4,
                 lr_scheduler='cosine_decay', decay_steps=100000,
                 warmup_steps=1000, warmup_factor=0.1, min_lr_factor=0.1):
        super().__init__()
        self.save_hyperparameters()

        dims = 63
        dims_per_vqvae = vqvae_config['nfeats']
        self.num_vqvaes = dims // dims_per_vqvae
        if dims % dims_per_vqvae != 0:
            self.num_vqvaes += 1  # Handle any remaining dimensions
        
        self.dims_per_vqvae = dims_per_vqvae
        self.vqvaes = nn.ModuleList([VQVae(**vqvae_config) for _ in range(self.num_vqvaes)])
        
        # Remove the single vqvae instance as we're using multiple now
        # self.vqvae = VQVae(**vqvae_config)
        self.losses_config = losses_config

        # Ensure learning rate is a float
        self.lr = float(lr)
        self.lr_scheduler = lr_scheduler
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor  # Initial LR multiplier during warmup
        self.min_lr_factor = min_lr_factor  # Minimum LR multiplier after decay

    def training_step(self, batch, batch_idx):
        # Get features from batch
        features = batch['features']  # Shape: [batch_size, max_seq_len, feature_dim]
        dim_ranges = batch['dim_ranges']
               
        # Initialize loss components
        total_recon_loss = 0
        total_commit_loss = 0
        total_perplexity = 0
        
        # Initialize reconstructed output tensor
        reconstructed = torch.zeros_like(features)
        
        # Process each slice of dimensions with its own VQVAE
        for i in range(self.num_vqvaes):
            start_idx = i * self.dims_per_vqvae
            end_idx = min((i + 1) * self.dims_per_vqvae, features.shape[-1])
            
            # Skip if we're past the feature dimensions
            if start_idx >= features.shape[-1]:
                continue
                
            # Extract feature slice for this VQVAE
            feature_slice = features[..., start_idx:end_idx]
            
            # Forward pass through the corresponding VQVAE
            reconstr_slice, commit_loss, perplexity = self.vqvaes[i](feature_slice)
            
            # Store reconstructed slice
            reconstructed[..., start_idx:end_idx] = reconstr_slice
            
            # Calculate reconstruction loss for this slice
            recon_loss = F.smooth_l1_loss(reconstr_slice, feature_slice)
            
            # Accumulate losses
            total_recon_loss += recon_loss
            total_commit_loss += commit_loss
            total_perplexity += perplexity
        
        # Average losses across all VQVAEs
        total_recon_loss /= self.num_vqvaes
        total_commit_loss /= self.num_vqvaes
        total_perplexity /= self.num_vqvaes
        
        # Total loss
        total_loss = (
            self.losses_config['lambda_feature'] * total_recon_loss + 
            self.losses_config['lambda_commit'] * total_commit_loss
        )

        # Log metrics with proper sync_dist setting
        # Main loss metrics - show in progress bar but only log epoch averages to wandb
        self.log('train/loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        # Detailed component losses - only log epoch averages to wandb
        self.log('train/recon_loss', total_recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/commit_loss', total_commit_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/perplexity', total_perplexity, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        
        # If you want more frequent logging for specific metrics, use logging_interval
        # Log learning rate at regular intervals (every N steps based on trainer.log_every_n_steps)
        if batch_idx % self.trainer.log_every_n_steps == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', current_lr, sync_dist=True, rank_zero_only=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Get features from batch
        features = batch['features']  # Shape: [batch_size, max_seq_len, feature_dim]

        
        # Initialize loss components
        total_recon_loss = 0
        total_commit_loss = 0
        total_perplexity = 0
        
        # Initialize reconstructed output tensor
        reconstructed = torch.zeros_like(features)
        
        # Process each slice of dimensions with its own VQVAE
        for i in range(self.num_vqvaes):
            start_idx = i * self.dims_per_vqvae
            end_idx = min((i + 1) * self.dims_per_vqvae, features.shape[-1])
            
            # Skip if we're past the feature dimensions
            if start_idx >= features.shape[-1]:
                continue
                
            # Extract feature slice for this VQVAE
            feature_slice = features[..., start_idx:end_idx]
            
            # Forward pass through the corresponding VQVAE
            reconstr_slice, commit_loss, perplexity = self.vqvaes[i](feature_slice)
            
            # Store reconstructed slice
            reconstructed[..., start_idx:end_idx] = reconstr_slice
            
            # Calculate reconstruction loss for this slice
            recon_loss = F.smooth_l1_loss(reconstr_slice, feature_slice)
            # Accumulate losses
            total_recon_loss += recon_loss
            total_commit_loss += commit_loss
            total_perplexity += perplexity
        
        # Average losses across all VQVAEs
        total_recon_loss /= self.num_vqvaes
        total_commit_loss /= self.num_vqvaes
        total_perplexity /= self.num_vqvaes
        # Total loss
        total_loss = (
            self.losses_config['lambda_feature'] * total_recon_loss + 
            self.losses_config['lambda_commit'] * total_commit_loss
        )

        # For validation, we typically want epoch-level statistics only
        self.log('val/loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/recon_loss', total_recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/commit_loss', total_commit_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/perplexity', total_perplexity, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)

        return total_loss

    def on_validation_epoch_end(self):
        # Get token usage stats
        all_stats = {}
        
        # Process token stats for each VQVAE separately
        for vqvae_idx, vqvae in enumerate(self.vqvaes):
            # Skip stats collection if we're past the valid VQVAEs
            if vqvae_idx >= self.num_vqvaes:
                continue
                
            # Check if we're using a ResVQVAE (with multiple quantizers) or regular VQVAE
            if hasattr(vqvae, 'quant_depth') and hasattr(vqvae, 'quantizers'):
                # ResVQVAE case with multiple quantizers
                for i in range(vqvae.quant_depth):
                    token_usage_stats = vqvae.quantizers[i].get_token_usage_stats()
                    
                    # Store stats for each layer with VQVAE index
                    for key, value in token_usage_stats.items():
                        stat_key = f'token_stats/vqvae{vqvae_idx}/{key}_layer{i}'
                        all_stats[stat_key] = value
                        # Log each stat separately
                        self.log(stat_key, value, sync_dist=True, rank_zero_only=True)
                
                # Calculate and log average stats across all layers for this VQVAE
                unique_tokens = [all_stats[f'token_stats/vqvae{vqvae_idx}/codebook/unique_tokens_layer{i}'] 
                                for i in range(vqvae.quant_depth)]
                usage_percent = [all_stats[f'token_stats/vqvae{vqvae_idx}/codebook/usage_percent_layer{i}'] 
                                for i in range(vqvae.quant_depth)]
                
                avg_unique_tokens = sum(unique_tokens) / vqvae.quant_depth
                avg_usage_percent = sum(usage_percent) / vqvae.quant_depth
                
                # Log average stats for this VQVAE
                self.log(f'token_stats/vqvae{vqvae_idx}/avg_unique_tokens', 
                        avg_unique_tokens, sync_dist=True, rank_zero_only=True)
                self.log(f'token_stats/vqvae{vqvae_idx}/avg_usage_percent', 
                        avg_usage_percent, sync_dist=True, rank_zero_only=True)
            
            elif hasattr(vqvae, 'quantizer'):
                # Regular VQVAE case with a single quantizer
                token_usage_stats = vqvae.quantizer.get_token_usage_stats()
                
                # Log stats for this VQVAE
                for key, value in token_usage_stats.items():
                    stat_key = f'token_stats/vqvae{vqvae_idx}/{key}'
                    self.log(stat_key, value, sync_dist=True, rank_zero_only=True)
        
        # Compute overall average statistics across all VQVAEs
        if self.num_vqvaes > 1:
            # Collect stats specifically for codebook usage metrics
            unique_tokens_all = []
            usage_percent_all = []
            
            for vqvae_idx in range(self.num_vqvaes):
                if hasattr(self.vqvaes[vqvae_idx], 'quantizer'):
                    # Simple case - get metrics directly
                    if f'token_stats/vqvae{vqvae_idx}/codebook/unique_tokens' in all_stats:
                        unique_tokens_all.append(all_stats[f'token_stats/vqvae{vqvae_idx}/codebook/unique_tokens'])
                        usage_percent_all.append(all_stats[f'token_stats/vqvae{vqvae_idx}/codebook/usage_percent'])
                elif hasattr(self.vqvaes[vqvae_idx], 'quantizers'):
                    # For ResVQVAE, use the already computed averages
                    if f'token_stats/vqvae{vqvae_idx}/avg_unique_tokens' in all_stats:
                        unique_tokens_all.append(all_stats[f'token_stats/vqvae{vqvae_idx}/avg_unique_tokens'])
                        usage_percent_all.append(all_stats[f'token_stats/vqvae{vqvae_idx}/avg_usage_percent'])
            
            # If we collected stats, compute and log the overall average
            if unique_tokens_all:
                overall_unique_tokens = sum(unique_tokens_all) / len(unique_tokens_all)
                overall_usage_percent = sum(usage_percent_all) / len(usage_percent_all)
                
                self.log('token_stats/overall_avg_unique_tokens', overall_unique_tokens, 
                        sync_dist=True, rank_zero_only=True)
                self.log('token_stats/overall_avg_usage_percent', overall_usage_percent, 
                        sync_dist=True, rank_zero_only=True)

    def configure_optimizers(self):
        # Only optimize VQVAE parameters, not MotionExtractor
        optimizer = torch.optim.Adam(
            self.vqvaes.parameters(),
            lr=self.lr,
            betas=(0.9, 0.99),
            weight_decay=0.0
        )

        if self.lr_scheduler == 'none':
            return optimizer
        elif self.lr_scheduler == 'cosine_decay':
            # Custom scheduler with warmup and cosine decay
            def warmup_cosine_decay(step):
                if step < self.warmup_steps:
                    # Linear warmup
                    alpha = float(step) / float(max(1, self.warmup_steps))
                    # Scale from warmup_factor to 1.0
                    return self.warmup_factor + alpha * (1.0 - self.warmup_factor)
                else:
                    # Cosine decay from 1.0 to min_lr_factor after warmup
                    progress = min(1.0, (step - self.warmup_steps) / float(max(1, self.decay_steps - self.warmup_steps)))
                    cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
                    # Scale from 1.0 to min_lr_factor
                    return self.min_lr_factor + cosine_decay * (1.0 - self.min_lr_factor)

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=warmup_cosine_decay
                ),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Unknown learning rate scheduler: {self.lr_scheduler}. Use 'cosine_decay' or 'none'.")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config):
    torch.set_float32_matmul_precision('medium')

    # Determine if this is the main process for distributed training
    is_main_process = (torch.cuda.is_available() and 
                      len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')) > 1 and
                      os.environ.get('LOCAL_RANK', '0') == '0') or True

    # Compose a concise run name
    lr_str = f"lr{config['learning_rate']}".replace('.', 'p')
    bs_str = f"bs{config['batch_size']}"
    epochs_str = f"e{config['max_epochs']}"
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_name = f"{cur_time}-{config['run_name']}-{lr_str}-{bs_str}-{epochs_str}"

    # Create run-specific directory structure
    run_dir = Path(config['output_path']) / run_name
    checkpoints_dir = run_dir / 'checkpoints'
    

    logger = None
    if is_main_process:
        logger = WandbLogger(
            project="liveportrait-tokenizer",
            name=run_name,
            config=config,
            log_model=False,
            save_dir=str(run_dir)
        )
        logger.log_hyperparams(config)
        pprint(f"Config: \n{config}")


    # Set up data
    # Get compute_stats from config, default to True if not specified
    compute_stats = config.get('compute_stats', True)
    # Create train dataset with normalization stats computation
    train_dataset = MotionDataset(
        config['data_path'], 
        split='train', 
        val_split=config['val_split'], 
        seed=config['seed'],
        compute_stats=compute_stats
    )
    
    # Create validation dataset, reusing the statistics from training set
    val_dataset = MotionDataset(
        config['data_path'], 
        split='val', 
        val_split=config['val_split'], 
        seed=config['seed'],
        compute_stats=False  # Don't compute stats for validation set
    )
    
    # If train dataset has computed stats, copy them to val dataset
    if train_dataset.mean is not None and train_dataset.std is not None and val_dataset.mean is None:
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std
        
    print(f"Loaded {len(train_dataset)} training pickle files and {len(val_dataset)} validation pickle files")

    print("Enabled features: ")
    for feat in sorted(config['feats_enabled']):
        if config['feats_enabled'][feat]['enabled']:
            print(feat)
    # Create a collate function with the configured max_seq_len
    collate_fn_with_max_len = lambda batch: collate_fn(batch, feats_enabled=config['feats_enabled'], max_seq_len=config['max_seq_len'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        persistent_workers=True if config['num_workers'] > 0 else False,
        collate_fn=collate_fn_with_max_len
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        persistent_workers=True if config['num_workers'] > 0 else False,
        collate_fn=collate_fn_with_max_len
    )

    # Set up model
    model = VQVAEModule(
        vqvae_config=config['vqvae'],
        losses_config=config['losses'],
        lr=config['learning_rate'],
        lr_scheduler=config['lr_scheduler']['type'],
        decay_steps=config['lr_scheduler']['decay_steps'],
        warmup_steps=config['lr_scheduler']['warmup_steps'],
        warmup_factor=config['lr_scheduler']['warmup_factor'],
        min_lr_factor=config['lr_scheduler']['min_lr_factor']
    )

    pretrained_path = config['vqvae'].get('pretrained_path', None)
    if pretrained_path is not None:
        print(f"Loading pretrained weights from {pretrained_path}")
        weights = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(weights['state_dict'])
        print(f"Successfully loaded pretrained weights from {pretrained_path}")

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename='checkpoint_{epoch:03d}',
        save_top_k=5,  # Save all checkpoints
        monitor='val/loss',
        every_n_epochs=config['checkpoint_frequency'],  # Use checkpoint frequency from config
        mode='min',
        save_weights_only=True,  # Only save model weights
        save_last=True,  # Don't save last checkpoint
        save_on_train_epoch_end=False,  # Only save on validation
    )

    callbacks = [checkpoint_callback]

    # Set up trainer with proper distributed training settings
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",
        default_root_dir=str(run_dir),  # Set the root directory for the run
        accumulate_grad_batches=1,
        log_every_n_steps=config['log_every_n_steps'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],  # Validate only every N epochs
        sync_batchnorm=True,
        enable_progress_bar=is_main_process,
        enable_model_summary=is_main_process,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = str(run_dir / f"final_{timestamp}.pth")
    if trainer.global_rank == 0:  # Only save on the main process
        # Save all VQVAEs parameters individually
        model_state = {}
        for i, vqvae in enumerate(model.vqvaes):
            # Get state dict for this VQVAE
            vqvae_state = vqvae.state_dict()
            # Add with a prefix to indicate which VQVAE it belongs to
            model_state[f'vqvae_{i}'] = vqvae_state
        
        # Add metadata about the VQVAEs
        model_state['num_vqvaes'] = model.num_vqvaes
        model_state['dims_per_vqvae'] = model.dims_per_vqvae
        
        torch.save(model_state, final_model_path)
        print(f"Saved final model to {final_model_path}")

    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Convert string paths to Path objects
    config['data_path'] = Path(config['data_path'])
    config['output_path'] = Path(config['output_path'])

    main(config)

# python train_tokenizer.py \
# --data_path dataset \
# --output_path models \
# --batch_size 32 \  # Now supports batching with variable sequence lengths
# --max_seq_len 300 \  # Maximum sequence length (longer sequences will be cropped)
# --num_workers 4 \
# --max_epochs 1 \
# --learning_rate 3e-4 \
# --lr_scheduler cosine \
# --warmup_steps 1000
