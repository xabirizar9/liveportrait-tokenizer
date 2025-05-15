import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from torch import Tensor
from torch.distributions.distribution import Distribution

from .quantizer import QuantizeEMAReset


class FCResidualBlock(nn.Module):
    """Fully-connected residual block with normalization and activation"""
    
    def __init__(self, dim, hidden_dim=None, activation='relu', norm=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim
            
        self.norm = norm
        
        if norm == "LN":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        else:  # Default to ReLU
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
            
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        
    def forward(self, x):
        x_orig = x
        
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.fc1(x)
        
        x = self.norm2(x)
        x = self.activation2(x)
        x = self.fc2(x)
        
        return x + x_orig


class FCResnet(nn.Module):
    """Stack of FC residual blocks"""
    
    def __init__(self, dim, depth, hidden_dim=None, activation='relu', norm=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = dim * 2
            
        blocks = [FCResidualBlock(dim, hidden_dim, activation=activation, norm=norm) 
                 for _ in range(depth)]
        
        self.model = nn.Sequential(*blocks)
        
    def forward(self, x):
        return self.model(x)


class FCEncoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 time_steps=64,  # Sequence length in input
                 down_steps=3,   # Number of downsampling steps
                 hidden_dim=512,
                 depth=2,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        self.time_steps = time_steps
        self.input_dim = input_emb_width
        self.down_steps = down_steps
        
        # Calculate compressed sequence length
        self.compressed_len = time_steps
        for _ in range(down_steps):
            self.compressed_len = self.compressed_len // 2
        
        # Input projection
        self.input_proj = nn.Linear(input_emb_width, hidden_dim)
        self.input_act = nn.ReLU()
        
        # Downsampling layers
        self.down_layers = nn.ModuleList()
        curr_seq_len = time_steps
        
        for i in range(down_steps):
            # Each layer halves the sequence length and processes features
            curr_seq_len = curr_seq_len // 2
            
            layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                FCResnet(hidden_dim, depth, hidden_dim=hidden_dim*2, activation=activation, norm=norm),
            )
            self.down_layers.append(layer)
        
        # Final projection to output embedding width
        self.output_proj = nn.Linear(hidden_dim, output_emb_width)
        
    def forward(self, x):
        # x shape: (batch_size, features, time_steps)
        batch_size, features, time_steps = x.shape
        
        # Reshape to (batch_size, time_steps, features)
        x = x.permute(0, 2, 1)
        
        # Initial projection
        x = self.input_proj(x)
        x = self.input_act(x)
        
        # Downsampling
        for i in range(self.down_steps):
            # Pairwise merge of adjacent timesteps
            x_pairs = []
            for j in range(0, x.size(1), 2):
                if j + 1 < x.size(1):
                    # Concatenate adjacent timesteps
                    pair = torch.cat([x[:, j], x[:, j+1]], dim=-1)
                    x_pairs.append(pair)
                else:
                    # If odd number of timesteps, pad the last one
                    pad = torch.zeros_like(x[:, j])
                    pair = torch.cat([x[:, j], pad], dim=-1)
                    x_pairs.append(pair)
                    
            # Stack pairs to form new sequence
            x = torch.stack(x_pairs, dim=1)
            
            # Apply FC layer and residual blocks
            x = self.down_layers[i](x)
        
        # Final projection
        x = self.output_proj(x)
        
        # Reshape back to (batch_size, features, compressed_time)
        x = x.permute(0, 2, 1)
        
        return x


class FCDecoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 time_steps=64,  # Original sequence length
                 down_steps=3,    # Number of upsampling steps (same as encoder's down_steps)
                 hidden_dim=512,
                 depth=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        self.time_steps = time_steps
        self.input_dim = input_emb_width
        self.down_steps = down_steps
        
        # Calculate compressed sequence length
        self.compressed_len = time_steps
        for _ in range(down_steps):
            self.compressed_len = self.compressed_len // 2
        
        # Input projection
        self.input_proj = nn.Linear(output_emb_width, hidden_dim)
        self.input_act = nn.ReLU()
        
        # Upsampling layers
        self.up_layers = nn.ModuleList()
        curr_seq_len = self.compressed_len
        
        for i in range(down_steps):
            # Each layer doubles the sequence length
            curr_seq_len = curr_seq_len * 2
            
            layer = nn.Sequential(
                FCResnet(hidden_dim, depth, hidden_dim=hidden_dim*2, activation=activation, norm=norm),
                nn.Linear(hidden_dim, hidden_dim * 2)  # Will be split into two timesteps
            )
            self.up_layers.append(layer)
        
        # Final projection
        self.output_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_act = nn.ReLU()
        self.output_proj2 = nn.Linear(hidden_dim, input_emb_width)
        
    def forward(self, x):
        # x shape: (batch_size, features, compressed_time)
        batch_size, features, compressed_time = x.shape
        
        # Reshape to (batch_size, compressed_time, features)
        x = x.permute(0, 2, 1)
        
        # Initial projection
        x = self.input_proj(x)
        x = self.input_act(x)
        
        # Upsampling
        for i in range(self.down_steps):
            # Apply FC layer and residual blocks
            x = self.up_layers[i](x)
            
            # Reshape to split features into two timesteps
            batch_size, seq_len, features = x.shape
            x = x.view(batch_size, seq_len, 2, features // 2)
            x = x.permute(0, 2, 1, 3).contiguous()
            x = x.view(batch_size, seq_len * 2, features // 2)
        
        # Final projections
        x = self.output_proj1(x)
        x = self.output_act(x)
        x = self.output_proj2(x)
        
        # Reshape back to (batch_size, features, time_steps)
        x = x.permute(0, 2, 1)
        
        return x


class FCVQVae(nn.Module):
    def __init__(self,
                 nfeats: int,
                 time_steps: int = 64,
                 code_num=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_steps=3,
                 hidden_dim=512,
                 depth=3,
                 norm=None,
                 activation: str = "relu",
                 apply_rotation_trick: bool = True,
                 use_quantization: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.code_dim = code_dim
        self.use_quantization = use_quantization
        self.time_steps = time_steps

        self.encoder = FCEncoder(
            input_emb_width=nfeats,
            output_emb_width=output_emb_width,
            time_steps=time_steps,
            down_steps=down_steps,
            hidden_dim=hidden_dim,
            depth=depth,
            activation=activation,
            norm=norm)

        self.decoder = FCDecoder(
            input_emb_width=nfeats,
            output_emb_width=output_emb_width,
            time_steps=time_steps,
            down_steps=down_steps,
            hidden_dim=hidden_dim,
            depth=depth,
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
                B, D, L = x_encoder.shape
                I = torch.eye(D, device=x_encoder.device).expand(B, L, D, D)
                rrt = torch.einsum('bdi,bdj->bdij', r, r)
                qet = torch.einsum('bdi,bdj->bdij', q_norm, e_norm)
                R = I - 2 * rrt + 2 * qet

                scaling = (x_quantized.norm(dim=1) / x_encoder.norm(dim=1)).unsqueeze(1)

                # Apply rotation and scaling as constants during backprop
                x_quantized = scaling * torch.einsum('bdij,bdj->bdi', R, x_encoder)
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

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in) # encode to latent space
        
        # Apply learnable scaling factor
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

        # decoder
        x_decoder = self.decoder(x_d)

        # Ensure output dimensions match the expected dimensions
        expected_length = self.time_steps
        if x_decoder.size(2) != expected_length:
            x_decoder = F.interpolate(x_decoder, size=expected_length, mode='linear', align_corners=False)

        x_out = self.postprocess(x_decoder)
        return x_out 