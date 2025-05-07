import torch
import numpy as np
import pandas as pd
import cv2
import imageio
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

from src.modules.motion_extractor import MotionExtractor
from src.live_portrait_wrapper import LivePortraitWrapper
from src.modules.vqvae import VQVae


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, split: str = 'train', val_split: float = 0.2, seed: int = 42):
        self.data_path = Path(data_path)
        self.video_dir = self.data_path / "train"
        self.split = split
        self.val_split = val_split
        self.seed = seed

        self.video_list = pd.read_csv(self.data_path / "videos_by_timestamp.csv")
        self.video_paths = [self.video_dir / f"{video_id}.mp4" for video_id in self.video_list['original_video_id'].unique()]

        # Split the dataset
        np.random.seed(seed)
        indices = np.random.permutation(len(self.video_paths))
        split_idx = int(len(indices) * (1 - val_split))

        if split == 'train':
            self.video_paths = [self.video_paths[i] for i in indices[:split_idx]]
        else:  # val
            self.video_paths = [self.video_paths[i] for i in indices[split_idx:]]

    def prepare_videos(self, imgs) -> torch.Tensor:
        """ construct the input as standard
        imgs: NxBxHxWx3, uint8
        """
        N, H, W, C = imgs.shape
        _imgs = imgs.reshape(N, H, W, C, 1)
        y = _imgs.astype(np.float32) / 255.
        y = np.clip(y, 0, 1)  # clip to 0~1
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        return y

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video = imageio.get_reader(video_path)
        frames = np.array([frame for frame in video])
        output = self.prepare_videos(frames)
        return output

    def __len__(self):
        return len(self.video_paths)


class VQVAEModule(pl.LightningModule):
    def __init__(self, nfeats=72, code_num=512, code_dim=512, output_emb_width=512,
                 down_t=3, stride_t=2, width=512, depth=3, dilation_growth_rate=3,
                 activation="relu", apply_rotation_trick=False, lr=1e-4,
                 lr_scheduler='cosine_decay', decay_steps=100000):
        super().__init__()
        self.save_hyperparameters()

        self.m_extr = MotionExtractor()
        self.m_extr.load_pretrained(init_path="pretrained_weights/liveportrait/base_models/motion_extractor.pth")

        # Freeze MotionExtractor parameters to prevent training
        for param in self.m_extr.parameters():
            param.requires_grad = False

        # Explicitly set model to eval mode to ensure inference behavior
        self.m_extr.eval()

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
            apply_rotation_trick=apply_rotation_trick
        )

        # Ensure learning rate is a float
        self.lr = float(lr)
        self.lr_scheduler = lr_scheduler
        self.decay_steps = decay_steps

    def training_step(self, batch, batch_idx):
        # Ensure MotionExtractor is in eval mode
        self.m_extr.eval()

        # Extract keypoints for each frame in the batch (only working for batch size 1)
        with torch.no_grad():
            kp_vid = torch.stack([self.m_extr(image)['kp'].squeeze(0) for image in batch[0]])
        kp_vid = kp_vid.unsqueeze(0)
        # Forward pass through VQVAE
        reconstr, commit_loss, perplexity = self.vqvae(kp_vid)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstr, kp_vid)

        # Total loss
        total_loss = recon_loss + commit_loss

        # Determine if we're using multiple GPUs
        sync_dist = True  # Always sync metrics in distributed training

        # Log metrics with proper sync_dist setting
        self.log('train_loss', total_loss, prog_bar=True, sync_dist=sync_dist, rank_zero_only=True)
        self.log('train_recon_loss', recon_loss, sync_dist=sync_dist, rank_zero_only=True)
        self.log('train_commit_loss', commit_loss, sync_dist=sync_dist, rank_zero_only=True)
        self.log('train_perplexity', perplexity, sync_dist=sync_dist, rank_zero_only=True)
        
        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, sync_dist=sync_dist, rank_zero_only=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Ensure MotionExtractor is in eval mode
        self.m_extr.eval()

        # Extract keypoints for each frame in the batch
        with torch.no_grad():
            kp_vid = torch.stack([self.m_extr(image)['kp'].squeeze(0) for image in batch[0]])
        kp_vid = kp_vid.unsqueeze(0)

        # Forward pass through VQVAE
        reconstr, commit_loss, perplexity = self.vqvae(kp_vid)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstr, kp_vid)

        # Total loss
        total_loss = recon_loss + commit_loss

        # Determine if we're using multiple GPUs
        sync_dist = True  # Always sync metrics in distributed training

        # Log metrics with proper sync_dist setting
        self.log('val_loss', total_loss, prog_bar=True, sync_dist=sync_dist, rank_zero_only=True)
        self.log('val_recon_loss', recon_loss, sync_dist=sync_dist, rank_zero_only=True)
        self.log('val_commit_loss', commit_loss, sync_dist=sync_dist, rank_zero_only=True)
        self.log('val_perplexity', perplexity, sync_dist=sync_dist, rank_zero_only=True)

        return total_loss

    def on_validation_epoch_end(self):
        # Log current learning rate
        self.vqvae.quantizer.get_token_usage_stats()
    

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
            # Custom cosine decay scheduler that only decays without cycling
            def cosine_decay(step):
                progress = min(1.0, step / self.decay_steps)
                cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
                return cosine_decay * (1 - 0.1) + 0.1  # Scale to [0.1, 1.0] range

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=cosine_decay
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
    code_num_str = f"c{config['code_num']}" if 'code_num' in config else ""
    run_name = f"vqvae-{lr_str}-{bs_str}{('-' + code_num_str) if code_num_str else ''}"

    # Optionally, allow user to override or append to run_name
    if 'run_name' in config and config['run_name']:
        run_name = f"{config['run_name']}-{lr_str}-{bs_str}{('-' + code_num_str) if code_num_str else ''}"

    config['run_name'] = run_name

    # Initialize wandb only on the main process
    if is_main_process:
        wandb.init(
            project="liveportrait-tokenizer",
            name=config['run_name'],
            config=config,
            dir=str(Path(config['output_path']) / "wandb")
        )

    # Set up data
    train_dataset = Dataset(config['data_path'], split='train', val_split=config['val_split'], seed=config['seed'])
    val_dataset = Dataset(config['data_path'], split='val', val_split=config['val_split'], seed=config['seed'])
    print(f"Loaded {len(train_dataset)} training videos and {len(val_dataset)} validation videos")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    # Set up model
    model = VQVAEModule(
        nfeats=config['nfeats'],
        code_num=config['code_num'],
        code_dim=config['code_dim'],
        output_emb_width=config['output_emb_width'],
        down_t=config['down_t'],
        stride_t=config['stride_t'],
        width=config['width'],
        depth=config['depth'],
        dilation_growth_rate=config['dilation_growth_rate'],
        activation=config['activation'],
        apply_rotation_trick=config['apply_rotation_trick'],
        lr=config['learning_rate'],
        lr_scheduler=config['lr_scheduler']['type'],
        decay_steps=config['lr_scheduler']['decay_steps']
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(config['output_path']) / 'checkpoints'),
        filename='vqvae-{epoch:02d}-step-{step}-loss-{train_loss:.4f}',
        save_top_k=3,
        save_last=True,
        every_n_train_steps=config['save_every_n_steps'],
        monitor='train_loss',
        mode='min'
    )

    val_checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(config['output_path']) / 'val_checkpoints'),
        filename='vqvae-val-{epoch:02d}-loss-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    callbacks = [checkpoint_callback, val_checkpoint_callback]

    # Set up wandb logger only on main process
    logger = None
    if is_main_process:
        logger = WandbLogger(
            project="liveportrait-tokenizer",
            name=config['run_name'],
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
        accumulate_grad_batches=4,
        log_every_n_steps=10,
        val_check_interval=100,
        limit_val_batches=100,
        sync_batchnorm=True,
        enable_progress_bar=is_main_process,
        enable_model_summary=is_main_process,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.get('run_name', 'vqvae')
    final_model_path = str(Path(config['output_path']) / f"{run_name}_final_{timestamp}.pth")
    if trainer.global_rank == 0:  # Only save on the main process
        model_state = model.vqvae.state_dict()
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
# --batch_size 1 \
# --num_workers 4 \
# --max_epochs 1 \
# --learning_rate 3e-4 \
# --lr_scheduler cosine \
# --warmup_steps 1000
