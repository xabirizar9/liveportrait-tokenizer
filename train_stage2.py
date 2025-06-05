import torch
import numpy as np
import os
import pytorch_lightning as pl
import wandb
import yaml
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
from src.motion_dataset import MotionDataset
from src.data_collator import collate_fn


class VQVAEStage2Module(pl.LightningModule):
    """Stage 2: Train VQ-VAE with frozen encoder and statistics-based codebook initialization"""
    def __init__(self, vqvae_config, losses_config, lr=5e-5,
                 lr_scheduler='cosine_decay', decay_steps=100000,
                 warmup_steps=1000, warmup_factor=0.1, min_lr_factor=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Create VQVAE with quantization enabled
        self.vqvae = VQVae(**vqvae_config)
        
        # Enable quantization for stage 2
        if not self.vqvae.use_quantization:
            self.vqvae.enable_quantization()
                
        self.codebook_initialized = False
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
        if self.codebook_initialized:
            return
            
        # For distributed training, synchronize initialization
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            
        # Initialize the codebook with statistics
        print("Initializing codebook with training set statistics...")
        
        # Get the training dataloader
        train_dataloader = self.trainer.train_dataloader
        
        # Initialize codebook with training set statistics
        self.vqvae.initialize_codebook_with_stats(
            train_dataloader,
            device=self.device
        )
        
        self.codebook_initialized = True
        
        # Ensure all processes have initialized the codebook before proceeding
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def training_step(self, batch, batch_idx):
        # Get features from batch
        features = batch['features']
        
        # Forward pass through VQVAE with quantization
        reconstr, commit_loss, perplexity = self.vqvae(features)
        
        # Calculate reconstruction loss
        recon_loss = F.smooth_l1_loss(reconstr, features)
        
        # Total loss (includes commitment loss)
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
        
        # Total loss (includes commitment loss)
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
        # Only optimize decoder and quantizer parameters, not encoder
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


def main(args):
    # Set precision
    torch.set_float32_matmul_precision('medium')

    # Load configuration
    config = load_config(args.config)
    
    # Convert string paths to Path objects
    config['data_path'] = Path(config['data_path'])
    config['output_path'] = Path(config['output_path'])
    

    # Determine if this is the main process for distributed training
    is_global_zero = (torch.cuda.is_available() and 
                    len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')) > 1 and
                    os.environ.get('LOCAL_RANK', '0') == '0') or True
    
    # Compose a concise run name
    lr_str = f"lr{config['learning_rate']}".replace('.', 'p')
    bs_str = f"bs{config['batch_size']}"
    epochs_str = f"e{config['max_epochs']}"
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_name = f"{cur_time}-stage2-{config.get('run_name', 'vqvae')}-{lr_str}-{bs_str}-{epochs_str}"

    # Create run-specific directory structure
    run_dir = Path(config['output_path']) / run_name
    checkpoints_dir = run_dir / 'checkpoints'
    
    # Create directories (only on main process)
    if is_global_zero:
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(exist_ok=True)

    # Initialize wandb (only on main process)
    logger = None
    if is_global_zero:
        logger = WandbLogger(
            project="liveportrait-tokenizer-stage2",
            name=run_name,
            config=config,
            log_model=False,
            save_dir=str(run_dir)
        )
        logger.log_hyperparams(config)
        print(f"Stage 2 Config: \n{config}")

    # Set up data
    compute_stats = config.get('compute_stats', True)
    train_dataset = MotionDataset(
        config['data_path'], 
        split='train', 
        val_split=config['val_split'], 
        seed=config['seed'],
        compute_stats=compute_stats
    )
    
    val_dataset = MotionDataset(
        config['data_path'], 
        split='val', 
        val_split=config['val_split'], 
        seed=config['seed'],
        compute_stats=False
    )
    
    # If train dataset has computed stats, copy them to val dataset
    if train_dataset.mean is not None and train_dataset.std is not None and val_dataset.mean is None:
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std
        
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

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

    # Set up stage 2 model
    model = VQVAEStage2Module(
        vqvae_config=config['vqvae'],
        losses_config=config['losses'],
        lr=config['learning_rate'],
        lr_scheduler=config['lr_scheduler']['type'],
        decay_steps=config['lr_scheduler']['decay_steps'],
        warmup_steps=config['lr_scheduler']['warmup_steps'],
        warmup_factor=config['lr_scheduler']['warmup_factor'],
        min_lr_factor=config['lr_scheduler']['min_lr_factor'],
    )

    # Load pretrained model if specified
    pretrained_path = config['vqvae'].get('pretrained_path', None)

    if pretrained_path:
        print(f"Loading pretrained model from {pretrained_path}")
        pretrained = torch.load(pretrained_path)
        
        # Load weights
        missing, unexpected = model.load_state_dict(pretrained['state_dict'])
        print(f"Loaded pretrained weights. Missing keys: {missing}, Unexpected keys: {unexpected}")
    else:
        print("No pretrained model provided")
        exit()

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        filename='stage2-checkpoint_{epoch:03d}',
        save_top_k=5,
        monitor='val/loss',
        every_n_epochs=config.get('checkpoint_frequency', 5),
        mode='min',
        save_weights_only=True,
        save_last=True,
        save_on_train_epoch_end=False,
    )

    callbacks = [checkpoint_callback]

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",
        default_root_dir=str(run_dir),
        accumulate_grad_batches=1,
        log_every_n_steps=config.get('log_every_n_steps', 50),
        check_val_every_n_epoch=config.get('check_val_every_n_epoch', 1),
        sync_batchnorm=True,
        enable_progress_bar=is_global_zero,
        enable_model_summary=is_global_zero,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = str(run_dir / f"stage2_final_{timestamp}.pth")
    if trainer.global_rank == 0:  # Only save on the main process
        # Only save VQVAE parameters
        torch.save(model.vqvae.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--pretrained_path", type=str, help="Path to pretrained VAE model")
    args = parser.parse_args()

    main(args) 