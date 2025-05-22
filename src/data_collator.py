import torch


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
        dim_ranges = {}
        
        # Track the current dimension index
        current_dim = 0

        for feat in sorted(feats_enabled):
            is_enabled = feats_enabled[feat]['enabled']

            if is_enabled:
                # Reshape the feature and get its flattened dimension
                if feat == 'exp_velocity' or feat == 'exp':
                    # Remove some of the features from exp_velocity (e.g. first 15 of the exp dim range)
                    reshaped_feat = sample[feat][..., :15, :].reshape(seq_len, -1)
                else:
                    reshaped_feat = sample[feat].reshape(seq_len, -1)
                
                feat_dim = reshaped_feat.shape[1]

                # Store the dimension range for this feature
                dim_ranges[feat] = (current_dim, current_dim + feat_dim)
                
                # Update the current dimension index
                current_dim += feat_dim
                
                # Add the reshaped feature to the list
                feats.append(reshaped_feat)
    
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
    
    return {'features': batched_features, "dim_ranges": dim_ranges}  # [batch_size, max_seq_len, feature_dim]