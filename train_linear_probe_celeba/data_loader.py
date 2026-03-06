"""
Data loading module for CelebA concept strengths and labels.
"""

import torch
import os.path as osp
from torch.utils.data import TensorDataset, DataLoader


def load_celeba_labels(celeba_root, split='train', attribute='Blond_Hair'):
    """
    Load CelebA attribute labels for a specific attribute.
    
    Args:
        celeba_root: Path to CelebA root directory
        split: 'train', 'val', or 'test'
        attribute: Attribute name (default: Blond_Hair)
    
    Returns:
        torch.Tensor of binary labels (0/1)
    """
    attr_file = osp.join(celeba_root, 'list_attr_celeba.txt')
    
    if not osp.exists(attr_file):
        raise FileNotFoundError(f"Attribute file not found: {attr_file}")
    
    # Parse attribute file
    with open(attr_file, 'r') as f:
        lines = f.readlines()
    
    # Get number of images and attribute names
    num_images = int(lines[0].strip())
    attr_names = lines[1].strip().split()
    
    # Find attribute index
    if attribute not in attr_names:
        raise ValueError(f"Attribute '{attribute}' not found. Available: {attr_names}")
    
    attr_idx = attr_names.index(attribute)
    
    # Load partition file to get split indices
    partition_file = osp.join(celeba_root, 'list_eval_partition.txt')
    with open(partition_file, 'r') as f:
        partitions = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # CelebA commonly stores: "<filename> <partition>"
            # Some variants store just: "<partition>"
            part = parts[-1]
            partitions.append(int(part))
    
    # Map split to partition number
    split_to_partition = {'train': 0, 'val': 1, 'test': 2}
    target_partition = split_to_partition[split]
    
    # Extract labels for this split
    labels = []
    for img_idx, partition in enumerate(partitions):
        if partition == target_partition:
            # Parse attribute line
            attr_line = lines[2 + img_idx].strip().split()
            attr_val = int(attr_line[attr_idx + 1])   # ← +1 to skip filename
            label = 1 if attr_val == 1 else 0
            labels.append(label)
    
    return torch.tensor(labels, dtype=torch.long)


def get_concept_strengths_loader(
    activations_path,
    celeba_root,
    split='train',
    batch_size=256,
    shuffle=True,
    attribute='Blond_Hair',
    device='cuda'
):
    """
    Create DataLoader with concept strengths and labels.
    
    Args:
        activations_path: Path to SAE activations file
        celeba_root: Path to CelebA root directory
        split: Data split
        batch_size: Batch size
        shuffle: Whether to shuffle data
        attribute: CelebA attribute to predict
        device: Device to load data on
    
    Returns:
        DataLoader with (concepts, labels) tuples
    """
    
    print(f"→ Loading concept strengths from: {activations_path}")
    concepts = torch.load(activations_path, map_location="cpu")
    if concepts.ndim == 3 and concepts.shape[1] == 1:
        concepts = concepts.squeeze(1)  # [N, 8192]
    print(f"✓ Concepts shape: {concepts.shape}")
    
    print(f"→ Loading CelebA labels for '{attribute}' attribute from: {celeba_root}")
    labels = load_celeba_labels(celeba_root, split=split, attribute=attribute)
    print(f"✓ Labels shape: {labels.shape}")
    
    # Check alignment
    assert concepts.shape[0] == labels.shape[0], \
        f"Mismatch: concepts {concepts.shape[0]} vs labels {labels.shape[0]}"
    
    # Create dataset and loader
    dataset = TensorDataset(concepts, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True if device == 'cuda' else False  # now valid, because tensors are CPU
    )
    
    print(f"✓ Created loader with {len(dataset)} samples\n")
    
    return loader, concepts.shape[0]