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

from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from pprint import pprint

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


from src.modules.vqvae import VQVae
from src.dataset import Dataset


def collate_fn(batch, max_seq_len=300):
    """
    Custom collate function for batching samples with fixed sequence length
    Args:
        batch: A list containing samples from dataset
        max_seq_len: Fixed sequence length to use (longer sequences will be cropped, shorter ones padded)
    Returns:
        Batched tensors with standardized sequence length
    """
    features_list = []

    feats_enabled = {
        'kp': True,
        'velocity': True,
        'exp': False,
        'x_s': True,
        't': True,
        'R': True,
        'scale': True,
    }
    
    for sample in batch:
        feats = []
        seq_len = sample['kp'].shape[0]

        if feats_enabled['kp']:
            kp = sample['kp'].reshape(seq_len, -1)  # [seq_len, 21*3]
            feats.append(kp)
        if feats_enabled['velocity']:
            velocity = sample['velocity'].reshape(seq_len, -1)  # [seq_len, 21*3]
            feats.append(velocity)
        if feats_enabled['exp']:
            exp = sample['exp'].reshape(seq_len, -1)  # [seq_len, 21*3]
            feats.append(exp)
        if feats_enabled['x_s']:
            x_s = sample['x_s'].reshape(seq_len, -1)  # [seq_len, 21*3]
            feats.append(x_s)
        if feats_enabled['t']:
            t = sample['t'].reshape(seq_len, -1)  # [seq_len, 3]
            feats.append(t)
        if feats_enabled['R']:
            R = sample['R'].reshape(seq_len, -1)  # [seq_len, 9]
            feats.append(R)
        if feats_enabled['scale']:
            scale = sample['scale'].reshape(seq_len, -1)  # [seq_len, 1]
            feats.append(scale)
    
        # Concatenate features
        features = torch.cat(feats, dim=1)  # [seq_len, N_feats]
        
        # Crop if longer than max_seq_len
        if seq_len > max_seq_len:
            features = features[:max_seq_len]
        
        # Pad if shorter than max_seq_len
        elif seq_len < max_seq_len:
            padding = torch.zeros((max_seq_len - seq_len, features.shape[1]), 
                                 dtype=features.dtype, device=features.device)
            features = torch.cat([features, padding], dim=0)
        
        features_list.append(features)
    
    # Stack along a new batch dimension
    batched_features = torch.stack(features_list)  # [batch_size, max_seq_len, feature_dim]
    
    return {'features': batched_features}  # [batch_size, max_seq_len, feature_dim]


class VQVAEModule(pl.LightningModule):
    def __init__(self, vqvae_config, losses_config, lr=1e-4,
                 lr_scheduler='cosine_decay', decay_steps=100000,
                 warmup_steps=1000, warmup_factor=0.1, min_lr_factor=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.vqvae = VQVae(**vqvae_config)
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
        # Forward pass through VQVAE
        reconstr, commit_loss, perplexity = self.vqvae(features)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.smooth_l1_loss(reconstr, features)
        velocity_loss = F.smooth_l1_loss(reconstr[:, :, 63:126], features[:, :, 63:126])
        # Total loss
        total_loss = (
            self.losses_config['lambda_feature'] * recon_loss + 
            self.losses_config['lambda_velocity'] * velocity_loss + 
            self.losses_config['lambda_commit'] * commit_loss
        )


        # Log metrics with proper sync_dist setting
        # Main loss metrics - show in progress bar but only log epoch averages to wandb
        self.log('train/loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/velocity_loss', velocity_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        
        # Detailed component losses - only log epoch averages to wandb
        self.log('train/recon_loss', recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/commit_loss', commit_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/perplexity', perplexity, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        
        # If you want more frequent logging for specific metrics, use logging_interval
        # Log learning rate at regular intervals (every N steps based on trainer.log_every_n_steps)
        if batch_idx % self.trainer.log_every_n_steps == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', current_lr, sync_dist=True, rank_zero_only=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Get features from batch
        features = batch['features']  # Shape: [batch_size, max_seq_len, feature_dim]

        # Forward pass through VQVAE
        reconstr, commit_loss, perplexity = self.vqvae(features)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.smooth_l1_loss(reconstr, features)
        velocity_loss = F.smooth_l1_loss(reconstr[:, :, 63:126], features[:, :, 63:126])
        # Total loss
        total_loss = (
            self.losses_config['lambda_feature'] * recon_loss + 
            self.losses_config['lambda_velocity'] * velocity_loss + 
            self.losses_config['lambda_commit'] * commit_loss
        )

        # For validation, we typically want epoch-level statistics only
        self.log('val/loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/recon_loss', recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/velocity_loss', velocity_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/commit_loss', commit_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/perplexity', perplexity, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)

        return total_loss

    def on_validation_epoch_end(self):
        # Get token usage stats
        token_usage_stats = self.vqvae.quantizer.get_token_usage_stats()
        
        # Log each stat separately instead of the whole dictionary
        for key, value in token_usage_stats.items():
            self.log(f'token_stats/{key}', value, sync_dist=True, rank_zero_only=True)

    def configure_optimizers(self):
        # Only optimize VQVAE parameters, not MotionExtractor
        optimizer = torch.optim.Adam(
            self.vqvae.parameters(),
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
    train_dataset = Dataset(
        config['data_path'], 
        split='train', 
        val_split=config['val_split'], 
        seed=config['seed'],
        compute_stats=compute_stats
    )
    
    # Create validation dataset, reusing the statistics from training set
    val_dataset = Dataset(
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

    # Create a collate function with the configured max_seq_len
    collate_fn_with_max_len = lambda batch: collate_fn(batch, max_seq_len=config['max_seq_len'])

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
        # Only save VQVAE parameters
        model_state = {k: v for k, v in model.state_dict().items() if k.startswith('vqvae.')}
        # Remove 'vqvae.' prefix from keys
        model_state = {k[6:]: v for k, v in model_state.items()}
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
