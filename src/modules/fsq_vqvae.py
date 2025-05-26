from typing import Union, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from .vqvae import Encoder, Decoder

from vector_quantize_pytorch import FSQ, ResidualFSQ

class FSQVAE(nn.Module):
    """
    VQ-VAE implementation that uses Group-Residual Finite Scalar Quantization (GRFSQ)
    instead of traditional vector quantization.
    """

    def __init__(self,
                 nfeats: int,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 norm=None,
                 activation: str = "relu",
                 pretrained_path: str = None,
                 levels: List[int] = [5, 5, 5, 5],
                 num_quantizers: int = 1,
                 use_quantization: bool = True,
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

        self.decoder = Decoder(
            nfeats,
            output_emb_width,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm)

        # Use the Group-Residual FSQ quantizer instead of QuantizeEMAReset
        levels = [levels[0]] * output_emb_width
        print(f"FSQ Config:")
        if use_quantization:
            print(f"Levels: {levels}")
            print(f"Output Emb Width: {output_emb_width}")
            print(f"Num Quantizers: {num_quantizers}")

            if num_quantizers == 1:
                print("Using FSQ")
                self.quantizer = FSQ(
                    levels=levels,
                    dim=output_emb_width,
                    preserve_symmetry=True,
                    return_indices=True
                )
            else:
                print("Using ResidualFSQ")
                self.quantizer = ResidualFSQ(
                    levels=levels,
                    num_quantizers=num_quantizers,
                    dim=output_emb_width,
                    preserve_symmetry=True,
                    return_indices=False
                )
        else:   
            print(f"Quantization disabled")

        
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor):
        # Converting to (bs, Jx3, T)
        x_in = self.preprocess(features)

        # Store original input length for exact reconstruction later
        original_length = x_in.size(2)

        # Encode -> (bs, C, T)
        x_encoder = self.encoder(x_in)

        # Converting back to (bs, T, C)
        x_encoder = self.postprocess(x_encoder)
        
        # Quantizer requires (bs, T, C, d)
        if self.use_quantization:
            x_quantized, _ = self.quantizer(x_encoder)
        else:
            x_quantized = x_encoder

        # Converting back to (bs, Jx3, T)
        x_quantized = self.preprocess(x_quantized)

        x_decoder = self.decoder(x_quantized)

        # Ensure output dimensions match input dimensions
        if x_decoder.size(2) != original_length:
            x_decoder = F.interpolate(x_decoder, size=original_length, mode='linear', align_corners=False)

        x_out = self.postprocess(x_decoder)

        return x_out


    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:
        """
        Encode inputs to latent representations.
        For FSQ, this returns the indices from the quantization process.
        """
        B, _, T = features.shape

        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in) # encode to latent space

        x_encoder = self.postprocess(x_encoder)
        
        # Apply FSQ and get quantized vectors
        x_quantized, indices = self.quantizer(x_encoder)
        
        # If indices were not returned (return_indices=False), calculate them
        if indices is None:
            print("Indices were not returned, calculating them")
            # Reshape for codes_to_indices if needed
            N, T, C = x_quantized.shape
            # Calculate indices directly from quantized vectors
            indices = self.quantizer.codes_to_indices(x_quantized.reshape(-1, self.output_emb_width))
            indices = indices.reshape(N, T, 1)  # Reshape back to [N, T, 1]
        
        # Ensure proper shape for returning
        N, T, C = x_encoder.shape if len(x_encoder.shape) == 3 else (B, T, self.output_emb_width)
        indices = indices.view(N, T, -1)  # Reshape to [N, T, flattened_indices]
        
        print(f"Indices shape: {indices.shape}")
    
        # Return indices and None for distribution (to match original interface)
        return indices

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
