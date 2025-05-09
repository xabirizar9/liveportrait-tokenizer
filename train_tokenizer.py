import torch
import numpy as np
import pandas as pd
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import torch.nn.functional as F
import torch.distributed
import yaml
from pathlib import Path
from argparse import ArgumentParser
import signal
import sys
from datetime import datetime

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
    
    for sample in batch:
        kp = sample['kp']
        exp = sample['exp']
        x_s = sample['x_s']
        t = sample['t']
        R = sample['R']
        scale = sample['scale']
        
        # Get sequence length from the first dimension of kp
        seq_len = kp.shape[0]
        
        # Reshape to (seq_len, -1)
        kp = kp.reshape(seq_len, -1)  # [seq_len, 21*3]
        exp = exp.reshape(seq_len, -1)  # [seq_len, 21*3]
        x_s = x_s.reshape(seq_len, -1)  # [seq_len, 21*3]
        t = t.reshape(seq_len, -1)  # [seq_len, 3]
        R = R.reshape(seq_len, -1)  # [seq_len, 9]
        scale = scale.reshape(seq_len, -1)  # [seq_len, 1]
        # Concatenate features
        features = torch.cat([kp, exp, x_s, t, R, scale], dim=1)  # [seq_len, 201]
        
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
    def __init__(self, nfeats=63, code_num=512, code_dim=512, output_emb_width=512,
                 down_t=3, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                 activation="relu", apply_rotation_trick=False, use_quantization=True, lr=1e-4,
                 lr_scheduler='cosine_decay', decay_steps=100000,
                 warmup_steps=1000, warmup_factor=0.1, min_lr_factor=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.vqvae = VQVae(
            nfeats=nfeats,
            code_num=code_num,
            code_dim=code_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            apply_rotation_trick=apply_rotation_trick,
            use_quantization=use_quantization,
        )

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
        recon_loss = F.mse_loss(reconstr, features)
        
        # Total loss
        total_loss = recon_loss + commit_loss


        # Log metrics with proper sync_dist setting
        # Main loss metrics - show in progress bar but only log epoch averages to wandb
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        
        # Detailed component losses - only log epoch averages to wandb
        self.log('train_recon_loss', recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train_commit_loss', commit_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('train_perplexity', perplexity, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        
        # If you want more frequent logging for specific metrics, use logging_interval
        # Log learning rate at regular intervals (every N steps based on trainer.log_every_n_steps)
        if batch_idx % self.trainer.log_every_n_steps == 0:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, sync_dist=True, rank_zero_only=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Get features from batch
        features = batch['features']  # Shape: [batch_size, max_seq_len, feature_dim]

        # Forward pass through VQVAE
        reconstr, commit_loss, perplexity = self.vqvae(features)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstr, features)

        # Total loss
        total_loss = recon_loss + commit_loss

        # For validation, we typically want epoch-level statistics only
        self.log('val_loss', total_loss, prog_bar=True, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val_commit_loss', commit_loss, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)
        self.log('val_perplexity', perplexity, sync_dist=True, rank_zero_only=True, on_step=False, on_epoch=True)

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

    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
    
    # Override config num_gpus with actual available GPUs
    config['num_gpus'] = num_gpus

    # Initialize wandb only on the main process
    if torch.cuda.is_available() and num_gpus > 1:
        is_main_process = (os.environ.get('LOCAL_RANK', '0') == '0')
    else:
        is_main_process = True

    # Compose a concise run name
    lr_str = f"lr{config['learning_rate']}".replace('.', 'p')
    bs_str = f"bs{config['batch_size']}"
    epochs_str = f"e{config['max_epochs']}"
    tokens_str = f"t{config['vqvae']['code_num']}"
    dim_str = f"d{config['vqvae']['code_dim']}"
    
    # Add quantization status to run name
    use_quantization = config['vqvae'].get('use_quantization', True)
    quant_str = "vq" if use_quantization else "vae"
    
    run_name = f"{config['run_name']}-{quant_str}-{lr_str}-{bs_str}-{epochs_str}-{tokens_str}-{dim_str}"

    # Initialize wandb only on the main process
    if is_main_process:
        wandb.init(
            project="liveportrait-tokenizer",
            name=run_name,
            config=config,
            dir=str(Path(config['output_path']) / "wandb")
        )

    # Set up data
    # Get compute_stats from config, default to True if not specified
    compute_stats = config.get('compute_stats', True)
    print(f"Config: \n{config}")
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
    max_seq_len = config.get('max_seq_len', 300)  # Default to 300 if not specified
    collate_fn_with_max_len = lambda batch: collate_fn(batch, max_seq_len=max_seq_len)

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
        nfeats=config['vqvae']['nfeats'],
        code_num=config['vqvae']['code_num'],
        code_dim=config['vqvae']['code_dim'],
        output_emb_width=config['vqvae']['output_emb_width'],
        down_t=config['vqvae']['down_t'],
        stride_t=config['vqvae']['stride_t'],
        width=config['vqvae']['width'],
        depth=config['vqvae']['depth'],
        dilation_growth_rate=config['vqvae']['dilation_growth_rate'],
        activation=config['vqvae']['activation'],
        apply_rotation_trick=config['vqvae']['apply_rotation_trick'],
        use_quantization=config['vqvae'].get('use_quantization', True),
        lr=config['learning_rate'],
        lr_scheduler=config['lr_scheduler']['type'],
        decay_steps=config['lr_scheduler']['decay_steps'],
        warmup_steps=config['lr_scheduler']['warmup_steps'],
        warmup_factor=config['lr_scheduler']['warmup_factor'],
        min_lr_factor=config['lr_scheduler']['min_lr_factor']
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(config['output_path']) / 'checkpoints'),
        filename='vqvae-{epoch:02d}-step-{step}',
        save_top_k=3,
        save_last=True,
        every_n_epochs=config['checkpoint_frequency'],  # Use checkpoint frequency from config
        monitor='val_loss',
        mode='min',
        save_weights_only=True  # Only save model weights
    )

    val_checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(config['output_path']) / 'val_checkpoints'),
        filename='vqvae-val-{epoch:02d}-step-{step}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_weights_only=True  # Only save model weights
    )

    callbacks = [checkpoint_callback, val_checkpoint_callback]

    # Set up wandb logger only on main process
    logger = None
    if is_main_process:
        logger = WandbLogger(
            project="liveportrait-tokenizer",
            name=run_name,
            log_model=False,
            save_dir=Path(config['output_path']) / "wandb"
        )
        logger.log_hyperparams(config)

    # Set up trainer with proper distributed training settings
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        logger=logger,
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        precision="bf16-mixed",
        default_root_dir=str(config['output_path']),
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
    final_model_path = str(Path(config['output_path']) / f"{run_name}_final_{timestamp}.pth")
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
