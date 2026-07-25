import torch

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    """
    Yields random (B, T) int64 tensors of inputs and shifted targets.
    
    Args:
        data: A 1D torch.Tensor of integer token IDs representing the full dataset.
        block_size: The sequence length T.
        batch_size: The number of sequences B in the batch.
        device: The device to move the tensors to (e.g., 'cpu', 'mps', 'cuda').
        
    Returns:
        x: Input tensor of shape (batch_size, block_size)
        y: Target tensor of shape (batch_size, block_size), shifted by 1 relative to x.
    """
    # Generate random starting indices for the sequences in the batch
    # The max index we can start from is len(data) - block_size - 1 
    # (since we need block_size elements for x, plus 1 more for the shifted y)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract inputs and targets using list comprehension and torch.stack
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    # Move to the desired device and ensure dtype is long (int64)
    x, y = x.to(device, dtype=torch.long), y.to(device, dtype=torch.long)
    
    return x, y
