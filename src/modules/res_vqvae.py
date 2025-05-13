# Partially from https://github.com/Mael-zys/T2M-GPT
# Adapted for residual quantization

from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.distributions.distribution import Distribution

from .vqvae import VQVae
from .quantizer import QuantizeEMAReset


class ResVQVae(VQVae):
    """Residual Vector Quantized Variational Autoencoder.
    
    This class extends the VQVae implementation with residual quantization capabilities,
    where the quantization error from previous steps is fed back into the quantizer
    to achieve more accurate representation.
    
    Each quantization layer uses its own codebook.
    """

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
                 apply_rotation_trick: bool = False,
                 use_quantization: bool = True,
                 quant_depth=3,
                 commitment_cost=0.25,
                 **kwargs) -> None:

        # Initialize parent class with all parameters except use_quantization set to True
        # We'll handle quantization ourselves with multiple codebooks
        super().__init__(
            nfeats=nfeats,
            code_num=code_num,
            code_dim=code_dim,
            output_emb_width=output_emb_width,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
            norm=norm,
            activation=activation,
            apply_rotation_trick=apply_rotation_trick,
            use_quantization=False,  # Set to False since we'll handle quantization ourselves
            **kwargs
        )
        
        # Residual quantization specific parameters
        self.quant_depth = quant_depth
        self.commitment_cost = commitment_cost
        
        # Create multiple quantizers, one for each residual layer
        self.quantizers = nn.ModuleList([
            QuantizeEMAReset(code_num, code_dim, mu=0.99)
            for _ in range(quant_depth)
        ])
        # Remove the single quantizer from parent class to avoid confusion
        if hasattr(self, 'quantizer'):
            delattr(self, 'quantizer')

    def forward(self, features: Tensor):
        """Forward pass with multi-codebook residual quantization."""
        # Preprocess
        x_in = self.preprocess(features)
        original_length = x_in.size(2)

        # Encode
        x_encoder = self.encoder(x_in)


        # Apply multi-codebook residual quantization
        z_detached = x_encoder.detach().clone()
        _residue = z_detached
        
        # Initialize variables to accumulate results
        total_commit_loss = 0
        total_perplexity = 0
        
        for d in range(self.quant_depth):
            # Use the appropriate quantizer for this layer
            _quantized, _commit_loss, _perplexity = self.quantizers[d](_residue)
            
            # Accumulate loss and perplexity
            total_commit_loss += _commit_loss
            total_perplexity += _perplexity
            
            if d == 0:
                z_hat = _quantized
            else:
                z_hat += _quantized
                
            # Calculate new residue for next iteration
            _residue = z_detached - z_hat
        
        # Average the perplexity across quantization steps
        perplexity = total_perplexity / self.quant_depth
        
        # Apply straight-through estimator for gradient flow
        x_quantized = x_encoder + (z_hat - x_encoder).detach()
               
        # Decode
        x_decoder = self.decoder(x_quantized)

        # Ensure output dimensions match input dimensions
        if x_decoder.size(2) != original_length:
            x_decoder = F.interpolate(x_decoder, size=original_length, mode='linear', align_corners=False)

        x_out = self.postprocess(x_decoder)

        return x_out, total_commit_loss, perplexity

    def encode(self, features: Tensor) -> Union[Tensor, List[Tensor]]:
        """Encode features to discrete codes from multiple codebooks."""
        N, T, _ = features.shape
        x_in = self.preprocess(features)
        z = self.encoder(x_in)  # encode to latent space


        # For multi-codebook residual quantization
        z_detached = z.detach().clone()
        _residue = z_detached
        
        # Store codes from each layer
        code_indices = []
        
        # Apply sequential residual quantization with separate codebooks
        for d in range(self.quant_depth):
            # Get quantized representation from this layer's quantizer
            _quantized, _, _ = self.quantizers[d](_residue)
            
            if d == 0:
                z_hat = _quantized
            else:
                z_hat += _quantized
            
            # Get the codes for this layer
            layer_z = _residue.permute(0, 2, 1).contiguous().view(-1, self.code_dim)
            layer_code_idx = self.quantizers[d].quantize(layer_z)
            layer_code_idx = layer_code_idx.view(N, -1)
            code_indices.append(layer_code_idx)
            
            # Calculate new residue for next iteration
            _residue = z_detached - z_hat
        
        # Return list of code indices from all codebooks
        return code_indices, None


    def decode(self, z: Union[Tensor, List[Tensor]]):
        """Decode indices from multiple codebooks to feature representation."""
        # Decode from multi-codebook representation
        batch_size = z[0].size(0)
        seq_len = z[0].size(1)
        
        # Initialize accumulator for quantized representations
        x_quantized = None
        
        # Process each codebook's codes
        for d, layer_code_idx in enumerate(z):
            # Get embeddings from this layer's quantizer
            layer_embeddings = self.quantizers[d].dequantize(layer_code_idx.reshape(-1))
            layer_embeddings = layer_embeddings.view(batch_size, seq_len, -1)
            
            # Accumulate embeddings
            if x_quantized is None:
                x_quantized = layer_embeddings
            else:
                x_quantized += layer_embeddings
        
        # Prepare for decoder
        x_d = x_quantized.permute(0, 2, 1).contiguous()

        # Expected output length after decoding
        token_length = x_d.size(2)
        expected_seq_len = token_length * (2 ** self.decoder.down_t)

        # Decode with the decoder network
        x_decoder = self.decoder(x_d)

        # Ensure consistent output length
        if x_decoder.size(2) != expected_seq_len:
            x_decoder = F.interpolate(x_decoder, size=expected_seq_len, mode='linear', align_corners=False)

        # Post-process to return to original format
        x_out = self.postprocess(x_decoder)
        return x_out
        
    def get_token_usage_stats(self):
        """Get token usage statistics from all codebooks."""
            
        stats = {}
        for i, quantizer in enumerate(self.quantizers):
            layer_stats = quantizer.get_token_usage_stats()
            # Add layer index to the stat names
            for key, value in layer_stats.items():
                stats[f"{key}_layer{i}"] = value
                
        # Add average stats across all layers
        unique_tokens = [stats[f"codebook/unique_tokens_layer{i}"] for i in range(self.quant_depth)]
        usage_percent = [stats[f"codebook/usage_percent_layer{i}"] for i in range(self.quant_depth)]
        
        stats["codebook/avg_unique_tokens"] = sum(unique_tokens) / self.quant_depth
        stats["codebook/avg_usage_percent"] = sum(usage_percent) / self.quant_depth
        
        return stats
