# Partially from https://github.com/Mael-zys/T2M-GPT

from typing import Union, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from .resnet import Resnet1D
from .quantizer import QuantizeEMAReset


class VQVae(nn.Module):

    def __init__(self,
                 nfeats: int,
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 apply_rotation_trick: bool = True,
                 use_quantization: bool = True,
                 pretrained_path: str = None,
                 **kwargs) -> None:

        super().__init__()
        self.code_dim = code_dim
        self.use_quantization = use_quantization

        self.encoder = Encoder(
            input_emb_width=nfeats,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            activation=activation,
            norm=norm)

        self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)

        self.apply_rotation_trick = apply_rotation_trick

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x
        
    @torch.no_grad()
    def initialize_codebook_kmeans(self, dataloader, device='cuda', max_samples=10000,
                                  return_centroids=False):
        """
        Initialize the codebook using KMeans on outputs from the trained encoder.
        
        Args:
            dataloader: DataLoader containing the training data
            device: Device to run inference on
            max_samples: Maximum number of samples to use for KMeans
            return_centroids: Whether to return the centroids
        """
        # Ensure we're in eval mode and quantization is disabled
        was_training = self.training
        self.eval()
        
        # Store original quantization state
        orig_quant_state = self.use_quantization
        self.use_quantization = False
        
        # Collect encoded vectors
        encoded_vectors = []
        sample_count = 0
        
        print("Collecting encoder outputs for KMeans clustering...")
        for batch in dataloader:
            if isinstance(batch, dict):
                features = batch['features'].to(device)
            else:
                features = batch.to(device)
                
            # Forward pass through encoder only
            N, T, _ = features.shape
            x_in = self.preprocess(features)
            x_encoder = self.encoder(x_in)
            
            # Flatten to (N*T, C) for clustering
            x_flat = x_encoder.permute(0, 2, 1).contiguous().view(-1, x_encoder.shape[1])
            
            encoded_vectors.append(x_flat)
            
            sample_count += x_flat.shape[0]
            if sample_count >= max_samples:
                break
                
        # Concatenate all collected vectors
        all_vectors = torch.cat(encoded_vectors, dim=0)
        
        # Use a subset if too many samples
        if all_vectors.shape[0] > max_samples:
            # Random sampling for KMeans
            indices = torch.randperm(all_vectors.shape[0])[:max_samples]
            all_vectors = all_vectors[indices]
            
        print(f"Collected {all_vectors.shape[0]} vectors for KMeans clustering")
        
        # Initialize codebook using KMeans
        inertia = self.quantizer.init_codebook_kmeans(all_vectors)
        
        # Restore original states
        if was_training:
            self.train()
        self.use_quantization = orig_quant_state
        
        print(f"Codebook initialization complete. KMeans inertia: {inertia:.4f}")
        
        if return_centroids:
            return inertia, self.quantizer.codebook
        else:
            return inertia

    def forward(self, features: Tensor):
        # Preprocess
        x_in = self.preprocess(features)

        # Store original input length for exact reconstruction later
        original_length = x_in.size(2)

        # Encode
        x_encoder = self.encoder(x_in)
        
        # Skip quantization if use_quantization is False
        if self.use_quantization:
            # quantization
            x_quantized, commit_loss, perplexity = self.quantizer(x_encoder)

            # Compute rotation matrix with detached gradients and apply rotation
            if self.apply_rotation_trick:
                with torch.no_grad():
                    # Normalize vectors for computing rotation
                    e_norm = F.normalize(x_encoder.detach(), dim=-1)
                    q_norm = F.normalize(x_quantized.detach(), dim=-1)

                # Compute r = (e + q)/||e + q|| for Householder reflection
                r = (e_norm + q_norm)
                r = F.normalize(r, dim=-1)

                # Compute rotation matrix R = I - 2rr^T + 2qe^T
                B, L, D = x_encoder.shape
                I = torch.eye(D, device=x_encoder.device).expand(B, L, D, D)
                rrt = torch.einsum('bli,blj->blij', r, r)
                qet = torch.einsum('bli,blj->blij', q_norm, e_norm)
                R = I - 2 * rrt + 2 * qet

                scaling = (x_quantized.norm(dim=-1) / x_encoder.norm(dim=-1)).unsqueeze(-1)

                # Apply rotation and scaling as constants during backprop
                x_quantized = scaling * torch.einsum('blij,blj->bli', R, x_encoder)
        else:
            # Skip quantization - use encoder output directly
            x_quantized = x_encoder
            commit_loss = torch.tensor(0.0, device=x_encoder.device)
            perplexity = torch.tensor(0.0, device=x_encoder.device)

        # decoder
        x_decoder = self.decoder(x_quantized)

        # Ensure output dimensions match input dimensions
        if x_decoder.size(2) != original_length:
            x_decoder = F.interpolate(x_decoder, size=original_length, mode='linear', align_corners=False)

        x_out = self.postprocess(x_decoder)

        return x_out, commit_loss, perplexity

    def freeze_encoder(self):
        """Freeze the encoder parameters for stage 2 training"""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen")
        
    def unfreeze_encoder(self):
        """Unfreeze the encoder parameters if needed"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder parameters unfrozen")
        
    def enable_quantization(self):
        """Enable VQ for stage 2 training"""
        self.use_quantization = True
        print("Quantization enabled")
        
    def disable_quantization(self):
        """Disable VQ for stage 1 training"""
        self.use_quantization = False
        print("Quantization disabled")

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in) # encode to latent space
        
        x_encoder = self.postprocess(x_encoder) # permutation
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)

        if self.use_quantization:
            code_idx = self.quantizer.quantize(x_encoder) # quantize to codebook
            code_idx = code_idx.view(N, -1)
        else:
            # Return the raw encoder output when not using quantization
            # Pack the continuous representation as if it were discrete codes
            code_idx = x_encoder.view(N, -1)

        # latent, dist
        return code_idx, None

    def decode(self, z: Tensor):
        if self.use_quantization:
            x_d = self.quantizer.dequantize(z)
            x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        else:
            # In non-quantized mode, reshape the latent representation directly
            # Assume z is already in the correct format (N, T*C)
            N = z.size(0)
            x_d = z.view(N, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # Expected output length after decoding (assuming each token represents a timestep)
        # This is an approximation - we multiply by stride_t^down_t since that's the theoretical downsampling
        token_length = x_d.size(2)
        expected_seq_len = token_length * (2 ** self.decoder.down_t)

        # decoder
        x_decoder = self.decoder(x_d)

        # Ensure consistent output length
        if x_decoder.size(2) != expected_seq_len:
            x_decoder = F.interpolate(x_decoder, size=expected_seq_len, mode='linear', align_corners=False)

        x_out = self.postprocess(x_decoder)
        return x_out


class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=2,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        # Simplified to single branch architecture
        blocks1 = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks1.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks1.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate,
                        activation=activation, norm=norm),
            )
            blocks1.append(block)
        blocks1.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.branch1 = nn.Sequential(*blocks1)

    def forward(self, x):
        return self.branch1(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        # Store parameters for interpolation and dimension calculation
        self.down_t = down_t
        self.stride_t = stride_t

        # Simplified to single branch architecture
        blocks1 = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks1.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks1.append(nn.ReLU())

        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate,
                        reverse_dilation=True, activation=activation,
                        norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks1.append(block)

        blocks1.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks1.append(nn.ReLU())
        blocks1.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.branch1 = nn.Sequential(*blocks1)

    def forward(self, x):
        return self.branch1(x)
