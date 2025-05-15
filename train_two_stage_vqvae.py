"""
Two-Stage VQ-VAE Training Script

This script implements a two-stage training process for VQ-VAE:
1. First stage: Train a regular VAE without quantization for 200 epochs
2. Second stage: Freeze the encoder, initialize the codebook with K-means, and train the quantizer and decoder
"""

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
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.modules.vqvae import VQVae
from src.dataset import Dataset


def collate_fn(batch, feats_enabled, max_seq_len=300):
    """
    Custom collate function for batching samples with fixed sequence length
    Args:
        batch: A list containing samples from dataset
        max_seq_len: Fixed sequence length to use (longer sequences will be cropped, shorter ones padded)
    Returns:
        Batched tensors with standardized sequence length
    """
    features_list = []

    for sample in batch:
        feats = []
        seq_len = sample['kp'].shape[0]

        for feat in feats_enabled:
            if feats_enabled[feat]['enabled']:
                feats.append(sample[feat].reshape(seq_len, -1))
    
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


class VQVAEStage1Module(pl.LightningModule):
    """Stage 1: Regular VAE (no quantization)"""
    def __init__(self, vqvae_config, losses_config, lr=1e-4,
                 lr_scheduler='cosine_decay', decay_steps=100000,
                 warmup_steps=1000, warmup_factor=0.1, min_lr_factor=0.1):
        super().__init__()
        self.save_hyperparameters()

        # Create VQVAE with quantization disabled
        self.vqvae = VQVae(
            **vqvae_config,
            use_quantization=False
        )
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
        
        # Forward pass through VQVAE without quantization
        reconstr, commit_loss, perplexity = self.vqvae(features)
        
        # Calculate reconstruction loss
        recon_loss = F.smooth_l1_loss(reconstr, features)
        
        # For Stage 1, we only use reconstruction loss
        total_loss = self.losses_config['lambda_feature'] * recon_loss
        
        # Log metrics
        self.log('train/loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/recon_loss', recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        
        # Log learning rate at regular intervals
        if batch_idx % self.trainer.log_every_n_steps == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', current_lr, sync_dist=True, rank_zero_only=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Get features from batch
        features = batch['features']
        
        # Forward pass through VQVAE without quantization
        reconstr, commit_loss, perplexity = self.vqvae(features)
        
        # Calculate reconstruction loss
        recon_loss = F.smooth_l1_loss(reconstr, features)
        
        # For Stage 1, we only use reconstruction loss
        total_loss = self.losses_config['lambda_feature'] * recon_loss
        
        # Log metrics
        self.log('val/loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/recon_loss', recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
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


class VQVAEStage2Module(pl.LightningModule):
    """Stage 2: VQ-VAE training with frozen encoder and KMeans codebook initialization"""
    def __init__(self, vqvae_module, losses_config, lr=5e-5,
                 lr_scheduler='cosine_decay', decay_steps=100000,
                 warmup_steps=1000, warmup_factor=0.1, min_lr_factor=0.1,
                 kmeans_initialized=False):
        super().__init__()
        self.save_hyperparameters(ignore=['vqvae_module'])
        
        # Transfer the VQVAE model from stage 1
        self.vqvae = vqvae_module.vqvae
        
        # Enable quantization for stage 2
        self.vqvae.enable_quantization()
        
        # Freeze encoder
        self.vqvae.freeze_encoder()
        
        self.kmeans_initialized = kmeans_initialized
        self.losses_config = losses_config
        
        # Ensure learning rate is a float
        self.lr = float(lr)
        self.lr_scheduler = lr_scheduler
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.min_lr_factor = min_lr_factor

    def on_train_start(self):
        # Skip if already initialized
        if self.kmeans_initialized:
            return
            
        # Get the training dataloader for KMeans initialization
        train_dataloader = self.trainer.train_dataloader
        
        # Initialize codebook with KMeans
        print("Initializing codebook with KMeans clustering...")
        self.vqvae.initialize_codebook_kmeans(
            train_dataloader, 
            device=self.device
        )
        # Save the KMeans centroids to a file
        torch.save(self.vqvae.quantizer.codebook, f"{self.trainer.default_root_dir}/kmeans_centroids.pt")
        self.kmeans_initialized = True
        print("KMeans initialization complete!")

    def training_step(self, batch, batch_idx):
        # Get features from batch
        features = batch['features']
        
        # Forward pass through VQVAE with quantization
        reconstr, commit_loss, perplexity = self.vqvae(features)
        
        # Calculate reconstruction loss
        recon_loss = F.smooth_l1_loss(reconstr, features)
        
        # Total loss (now includes commitment loss)
        total_loss = (
            self.losses_config['lambda_feature'] * recon_loss + 
            self.losses_config['lambda_commit'] * commit_loss
        )
        
        # Log metrics
        self.log('train/loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/recon_loss', recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/commit_loss', commit_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train/perplexity', perplexity, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        
        # Log learning rate at regular intervals
        if batch_idx % self.trainer.log_every_n_steps == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('train/learning_rate', current_lr, sync_dist=True, rank_zero_only=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Get features from batch
        features = batch['features']
        
        # Forward pass through VQVAE with quantization
        reconstr, commit_loss, perplexity = self.vqvae(features)
        
        # Calculate reconstruction loss
        recon_loss = F.smooth_l1_loss(reconstr, features)
        
        # Total loss (now includes commitment loss)
        total_loss = (
            self.losses_config['lambda_feature'] * recon_loss + 
            self.losses_config['lambda_commit'] * commit_loss
        )
        
        # Log metrics
        self.log('val/loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/recon_loss', recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/commit_loss', commit_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val/perplexity', perplexity, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)

        return total_loss

    def on_validation_epoch_end(self):
        # Get token usage stats
        if hasattr(self.vqvae, 'quantizer'):
            token_usage_stats = self.vqvae.quantizer.get_token_usage_stats()
            
            # Log stats
            for key, value in token_usage_stats.items():
                stat_key = f'token_stats/{key}'
                self.log(stat_key, value, sync_dist=True, rank_zero_only=True)

    def configure_optimizers(self):
        # Only optimize decoder and quantizer parameters
        trainable_params = list(self.vqvae.decoder.parameters()) + list(self.vqvae.quantizer.parameters())
        
        optimizer = torch.optim.Adam(
            trainable_params,
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
    lr1_str = f"lr1_{config['stage1_learning_rate']}".replace('.', 'p')
    lr2_str = f"lr2_{config['stage2_learning_rate']}".replace('.', 'p')
    bs_str = f"bs{config['batch_size']}"
    epochs_str = f"e1_{config['stage1_epochs']}_e2_{config['stage2_epochs']}"
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_name = f"{cur_time}-{config['run_name']}-{lr1_str}-{lr2_str}-{bs_str}-{epochs_str}"

    # Create run-specific directory structure
    run_dir = Path(config['output_path']) / run_name
    stage1_checkpoints_dir = run_dir / 'stage1_checkpoints'
    stage2_checkpoints_dir = run_dir / 'stage2_checkpoints'
    
    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)
    stage1_checkpoints_dir.mkdir(exist_ok=True)
    stage2_checkpoints_dir.mkdir(exist_ok=True)

    # Initialize wandb for stage 1
    logger_stage1 = None
    if is_main_process:
        logger_stage1 = WandbLogger(
            project="liveportrait-tokenizer-stage1",
            name=f"{run_name}-stage1",
            config=config,
            log_model=False,
            save_dir=str(run_dir)
        )
        logger_stage1.log_hyperparams(config)
        print(f"Stage 1 Config: \n{config}")

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

    # ==========================
    # STAGE 1: Train VAE without quantization
    # ==========================
    print("\n" + "="*50)
    print("STAGE 1: Training VAE without quantization")
    print("="*50 + "\n")

    # Set up stage 1 model
    stage1_model = VQVAEStage1Module(
        vqvae_config=config['vqvae'],
        losses_config=config['losses'],
        lr=config['stage1_learning_rate'],
        lr_scheduler=config['lr_scheduler']['type'],
        decay_steps=config['lr_scheduler']['decay_steps'],
        warmup_steps=config['lr_scheduler']['warmup_steps'],
        warmup_factor=config['lr_scheduler']['warmup_factor'],
        min_lr_factor=config['lr_scheduler']['min_lr_factor']
    )

    # Set up stage 1 callbacks
    stage1_checkpoint_callback = ModelCheckpoint(
        dirpath=str(stage1_checkpoints_dir),
        filename='stage1_checkpoint_{epoch:03d}',
        save_top_k=5,
        monitor='val/loss',
        every_n_epochs=config['checkpoint_frequency'],
        mode='min',
        save_weights_only=True,
        save_last=True,
        save_on_train_epoch_end=False,
    )

    callbacks_stage1 = [stage1_checkpoint_callback]

    # Set up stage 1 trainer
    trainer_stage1 = pl.Trainer(
        max_epochs=config['stage1_epochs'],
        callbacks=callbacks_stage1,
        logger=logger_stage1,
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",
        default_root_dir=str(run_dir),
        accumulate_grad_batches=1,
        log_every_n_steps=config['log_every_n_steps'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        sync_batchnorm=True,
        enable_progress_bar=is_main_process,
        enable_model_summary=is_main_process,
    )

    # Train stage 1 model
    trainer_stage1.fit(stage1_model, train_loader, val_loader)

    # Save stage 1 model
    stage1_final_path = str(run_dir / "stage1_final.pth")
    if trainer_stage1.global_rank == 0:
        # Only save VQVAE parameters
        model_state = {k: v for k, v in stage1_model.state_dict().items() if k.startswith('vqvae.')}
        # Remove 'vqvae.' prefix from keys
        model_state = {k[6:]: v for k, v in model_state.items()}
        torch.save(model_state, stage1_final_path)
        print(f"Saved stage 1 final model to {stage1_final_path}")

    # Finish wandb for stage 1
    if logger_stage1 is not None:
        wandb.finish()

    # ==========================
    # STAGE 2: Train VQ-VAE with frozen encoder and KMeans initialization
    # ==========================
    print("\n" + "="*50)
    print("STAGE 2: Training VQ-VAE with frozen encoder and KMeans initialization")
    print("="*50 + "\n")

    # Initialize wandb for stage 2
    logger_stage2 = None
    if is_main_process:
        logger_stage2 = WandbLogger(
            project="liveportrait-tokenizer-stage2",
            name=f"{run_name}-stage2",
            config=config,
            log_model=False,
            save_dir=str(run_dir)
        )
        logger_stage2.log_hyperparams(config)
        print(f"Stage 2 Config: \n{config}")

    # Set up stage 2 model
    stage2_model = VQVAEStage2Module(
        vqvae_module=stage1_model,
        losses_config=config['losses'],
        lr=config['stage2_learning_rate'],
        lr_scheduler=config['lr_scheduler']['type'],
        decay_steps=config['lr_scheduler']['decay_steps'],
        warmup_steps=config['lr_scheduler']['warmup_steps'],
        warmup_factor=config['lr_scheduler']['warmup_factor'],
        min_lr_factor=config['lr_scheduler']['min_lr_factor']
    )

    # Set up stage 2 callbacks
    stage2_checkpoint_callback = ModelCheckpoint(
        dirpath=str(stage2_checkpoints_dir),
        filename='stage2_checkpoint_{epoch:03d}',
        save_top_k=5,
        monitor='val/loss',
        every_n_epochs=config['checkpoint_frequency'],
        mode='min',
        save_weights_only=True,
        save_last=True,
        save_on_train_epoch_end=False,
    )

    callbacks_stage2 = [stage2_checkpoint_callback]

    # Set up stage 2 trainer
    trainer_stage2 = pl.Trainer(
        max_epochs=config['stage2_epochs'],
        callbacks=callbacks_stage2,
        logger=logger_stage2,
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",
        default_root_dir=str(run_dir),
        accumulate_grad_batches=1,
        log_every_n_steps=config['log_every_n_steps'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
        sync_batchnorm=True,
        enable_progress_bar=is_main_process,
        enable_model_summary=is_main_process,
    )

    # Train stage 2 model
    trainer_stage2.fit(stage2_model, train_loader, val_loader)

    # Save stage 2 final model
    stage2_final_path = str(run_dir / "stage2_final.pth")
    if trainer_stage2.global_rank == 0:
        # Only save VQVAE parameters
        model_state = {k: v for k, v in stage2_model.state_dict().items() if k.startswith('vqvae.')}
        # Remove 'vqvae.' prefix from keys
        model_state = {k[6:]: v for k, v in model_state.items()}
        torch.save(model_state, stage2_final_path)
        print(f"Saved stage 2 final model to {stage2_final_path}")

    # Finish wandb for stage 2
    if logger_stage2 is not None:
        wandb.finish()

    print("\nTwo-stage training complete!")
    print(f"Stage 1 model saved to: {stage1_final_path}")
    print(f"Stage 2 model saved to: {stage2_final_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/two_stage.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Convert string paths to Path objects
    config['data_path'] = Path(config['data_path'])
    config['output_path'] = Path(config['output_path'])

    main(config) 