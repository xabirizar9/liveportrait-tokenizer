from typing import Union, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from .resnet import Resnet1D
from .finite_scalar_quantizer import GroupResidualFSQ

class FSQVAE(nn.Module):
    """
    VQ-VAE implementation that uses Group-Residual Finite Scalar Quantization (GRFSQ)
    instead of traditional vector quantization.
    """

    def __init__(self,
                 nfeats: int,
                 fsq_levels=[0.0, 0.33, 0.66, 1.0],
                 num_groups=8,
                 num_residual_quantizers=2,
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
        self.output_emb_width = output_emb_width
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

        # Use the Group-Residual FSQ quantizer instead of QuantizeEMAReset
        self.quantizer = GroupResidualFSQ(
            input_dim=output_emb_width,
            levels=fsq_levels,
            num_groups=num_groups,
            num_residual_quantizers=num_residual_quantizers
        )

        self.apply_rotation_trick = apply_rotation_trick
        
        # Store FSQ configuration for reference
        self.fsq_levels = fsq_levels
        self.num_groups = num_groups
        self.num_residual_quantizers = num_residual_quantizers

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        # Preprocess
        x_in = self.preprocess(features)

        # Store original input length for exact reconstruction later
        original_length = x_in.size(2)

        # Encode
        x_encoder = self.encoder(x_in)
        
        # Skip quantization if use_quantization is False
        if self.use_quantization:
            # Apply FSQ quantization
            x_quantized, indices, commit_loss = self.quantizer(x_encoder)
            
            # Calculate perplexity-like metric for consistency with original VQ-VAE
            # Number of unique levels used in each group/residual quantizer
            with torch.no_grad():
                stats = self.quantizer.get_token_usage_stats()
                perplexity = stats.get('codebook/usage_percent', 0)
                perplexity = torch.tensor(perplexity, device=x_encoder.device)

            # Apply rotation trick similar to original implementation
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
        """Enable quantization for stage 2 training"""
        self.use_quantization = True
        print("Quantization enabled")
        
    def disable_quantization(self):
        """Disable quantization for stage 1 training"""
        self.use_quantization = False
        print("Quantization disabled")

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:
        """
        Encode inputs to latent representations.
        For FSQ, this returns the indices from the quantization process.
        """
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in) # encode to latent space
        
        if self.use_quantization:
            # Apply FSQ and get the indices
            _, indices, _ = self.quantizer(x_encoder)
            
            # Reshape indices for output - we'll flatten the group and residual dimensions
            # Indices shape: [N*T, num_groups, num_residual_quantizers, group_dim]
            # We want to reshape to [N, T, num_groups * num_residual_quantizers * group_dim]
            N, C, T = x_encoder.shape
            indices = indices.view(N * T, -1)  # Flatten all indices dimensions
            indices = indices.view(N, T, -1)   # Reshape to [N, T, flattened_indices]
        else:
            # Return the raw encoder output when not using quantization
            x_encoder = self.postprocess(x_encoder)  # [N, T, C]
            indices = x_encoder.reshape(N, T, -1)    # Same shape but clarifying intent
            
        # Return indices and None for distribution (to match original interface)
        return indices, None

    def decode(self, z: Tensor):
        """
        Decode from latent representation.
        For FSQ, we'd typically need to process the indices, but for simplicity
        we'll assume z contains the quantized representation directly.
        """
        N, T, D = z.shape
        
        # Reshape to expected format for decoder [N, C, T]
        z_reshaped = z.permute(0, 2, 1).contiguous()
        
        # Expected output length after decoding
        expected_seq_len = T * (2 ** self.decoder.down_t)
        
        # Decode
        x_decoder = self.decoder(z_reshaped)
        
        # Ensure consistent output length
        if x_decoder.size(2) != expected_seq_len:
            x_decoder = F.interpolate(x_decoder, size=expected_seq_len, mode='linear', align_corners=False)
            
        # Post-process
        x_out = self.postprocess(x_decoder)
        return x_out


# Reuse the same Encoder and Decoder classes as in the original VQVAE
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