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

        for feat in feats_enabled:
            if feats_enabled[feat]['enabled']:
                feats.append(sample[feat].reshape(seq_len, -1))
    
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
    
    return {'features': batched_features}  # [batch_size, max_seq_len, feature_dim]