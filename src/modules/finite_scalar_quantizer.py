import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FSQQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ) module as described in VQTalker paper.
    
    FSQ quantizes each dimension to a set of fixed levels without explicit codebook lookup.
    """
    def __init__(self, levels):
        """
        Args:
            levels (list): Quantization levels for each dimension [l1, l2, ..., ld]
        """
        super().__init__()
        self.levels = levels
        self.register_buffer('levels_tensor', torch.tensor(levels, dtype=torch.float32))
    
    def forward(self, x):
        """
        Quantize input using FSQ
        
        Args:
            x (tensor): Input tensor to quantize
            
        Returns:
            tuple: (quantized output, indices)
        """
        # Scale input to [0, 1] range for quantization
        min_val = x.min(dim=-1, keepdim=True)[0]
        max_val = x.max(dim=-1, keepdim=True)[0]
        
        # Avoid division by zero
        denominator = max_val - min_val
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        
        x_scaled = (x - min_val) / denominator
        
        # Quantize each value to nearest level
        indices = torch.zeros_like(x, dtype=torch.long)
        x_quantized = torch.zeros_like(x)
        
        for i, level in enumerate(self.levels):
            # Find values closest to this level
            mask = torch.logical_and(
                x_scaled >= (level - 0.5/len(self.levels)),
                x_scaled < (level + 0.5/len(self.levels))
            )
            indices = torch.where(mask, torch.full_like(indices, i), indices)
            x_quantized = torch.where(mask, torch.full_like(x_quantized, level), x_quantized)
        
        # Rescale back to original range
        x_quantized = x_quantized * denominator + min_val
        
        x_quantized_st = x + (x_quantized - x).detach()
        
        return x_quantized_st, indices


class GroupResidualFSQ(nn.Module):
    """
    Group-Residual Finite Scalar Quantization (GRFSQ) as described in VQTalker paper.
    
    Combines Group VQ, Residual VQ, and FSQ techniques.
    """
    def __init__(self, input_dim, levels, num_groups, num_residual_quantizers):
        """
        Args:
            input_dim (int): Dimension of input features
            levels (list): Quantization levels for FSQ
            num_groups (int): Number of groups to split the input
            num_residual_quantizers (int): Number of residual quantization steps
        """
        super().__init__()
        self.input_dim = input_dim
        self.levels = levels
        self.num_groups = num_groups
        self.num_residual_quantizers = num_residual_quantizers
        
        # Ensure input dimension is divisible by num_groups
        assert input_dim % num_groups == 0, "Input dimension must be divisible by number of groups"
        self.group_dim = input_dim // num_groups
        
        # Create FSQ quantizers for each residual step
        self.fsq_quantizers = nn.ModuleList([
            FSQQuantizer(levels) for _ in range(num_residual_quantizers)
        ])
        
        # Register statistics buffer for token usage tracking
        self.register_buffer('val_token_usage', torch.zeros(
            num_groups, num_residual_quantizers, len(levels), dtype=torch.long))
    
    def forward(self, x):
        """
        Apply GRFSQ to input tensor
        
        Args:
            x (tensor): Input tensor of shape [batch_size, input_dim, sequence_length]
            
        Returns:
            tuple: (quantized tensor, indices)
        """
        batch_size, feature_dim, seq_len = x.shape
        assert feature_dim == self.input_dim, f"Expected input dimension {self.input_dim}, got {feature_dim}"
        
        # Reshape to [batch_size * seq_len, feature_dim]
        x_flat = x.permute(0, 2, 1).reshape(-1, feature_dim)
        
        # Split into groups: [batch_size * seq_len, num_groups, group_dim]
        groups = x_flat.view(-1, self.num_groups, self.group_dim)
        
        # Initialize quantized output and indices
        quantized_groups = torch.zeros_like(groups)
        indices = torch.zeros(
            groups.shape[0], self.num_groups, self.num_residual_quantizers, groups.shape[2],
            device=x.device, dtype=torch.long
        )
        
        # Process each group
        for g in range(self.num_groups):
            residual = groups[:, g]
            
            # Apply residual quantizers
            for r in range(self.num_residual_quantizers):
                z_quantized, z_indices = self.fsq_quantizers[r](residual)
                
                # Add quantized values to output
                quantized_groups[:, g] += z_quantized
                
                # Update residual
                residual = residual - z_quantized
                
                # Store indices
                indices[:, g, r] = z_indices
                
                # Track token usage during validation
                if not self.training:
                    with torch.no_grad():
                        for level_idx in range(len(self.levels)):
                            self.val_token_usage[g, r, level_idx] += (z_indices == level_idx).sum().item()
        
        # Reshape back to original dimensions
        x_quantized = quantized_groups.reshape(batch_size, seq_len, feature_dim).permute(0, 2, 1)
        
        # Compute quantization loss (MSE between input and quantized output)
        commit_loss = F.mse_loss(x, x_quantized.detach())
        
        # Straight-through estimator for backprop
        x_quantized_st = x + (x_quantized - x).detach()
        
        return x_quantized_st, indices, commit_loss
    
    def get_token_usage_stats(self):
        """
        Get token usage statistics and reset counters
        
        Returns:
            dict: Token usage statistics
        """
        with torch.no_grad():
            # Calculate statistics
            total_usage = self.val_token_usage.sum().item()
            if total_usage == 0:
                return {'codebook/usage_percent': 0}
            
            # Calculate percentage of levels used
            levels_used = (self.val_token_usage > 0).sum().item()
            total_levels = self.num_groups * self.num_residual_quantizers * len(self.levels)
            usage_percentage = (levels_used / total_levels) * 100
            
            stats = {
                'codebook/unique_levels': levels_used,
                'codebook/usage_percent': usage_percentage,
            }
            
            # Reset counters for next validation epoch
            self.val_token_usage.zero_()
            
            return stats 