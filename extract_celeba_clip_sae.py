"""
Extract CLIP features and SAE sparse activations for CelebA dataset.
Adapted from scripts/save_cc3m_features.py and scripts/save_concept_strengths.py
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import os.path as osp
from pprint import pprint

from dncbm.utils import common_init, get_img_model, get_sae_ckpt
from dncbm import arg_parser, config
from sparse_autoencoder import SparseAutoencoder


class CelebADataset(Dataset):
    """CelebA dataset loader - compatible with CLIP preprocessing"""
    
    def __init__(self, root_dir, preprocess_fn, split='train'):
        """
        Args:
            root_dir: Path to CelebA root directory containing:
                - img_align_celeba/ (directory with all images)
                - list_eval_partition.txt (file with image split assignments)
            preprocess_fn: Preprocessing function (e.g., CLIP preprocess)
            split: 'train', 'val', or 'test'
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / 'img_align_celeba'
        self.preprocess = preprocess_fn
        self.split = split
        
        # Validate directories exist
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        
        # Load split file
        split_file = self.root_dir / 'list_eval_partition.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        # Map split names to partition indices
        split_map = {'train': 0, 'val': 1, 'test': 2}
        partition_idx = split_map[split]
        
        # Read split file and collect image filenames
        self.image_files = []
        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name, partition = parts[0], int(parts[1])
                    if partition == partition_idx:
                        self.image_files.append(img_name)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found for split '{split}'")
        
        print(f"✓ Loaded {len(self.image_files)} images for '{split}' split from CelebA")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.img_dir / img_name
        
        try:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img)
            return img_tensor, idx
        except Exception as e:
            print(f"⚠ Error loading {img_path}: {e}")
            return None, idx


class CLIPSAEExtractor:
    """Extract CLIP features and SAE sparse activations"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # Load CLIP model
        print(f"\n→ Loading CLIP model: {args.img_enc_name}")
        self.model, self.preprocess = get_img_model(args)
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Load SAE checkpoint
        print(f"→ Loading SAE checkpoint from: {args.save_dir_sae_ckpts[args.modality]}")
        self.sae = self._load_sae()
        self.sae.eval()
        print("✓ SAE loaded successfully")
    
    def _load_sae(self):
        """Load SAE from checkpoint"""
        autoencoder_input_dim = self.args.autoencoder_input_dim_dict[
            self.args.ae_input_dim_dict_key[self.args.modality]
        ]
        n_learned_features = int(
            autoencoder_input_dim * self.args.expansion_factor
        )
        
        sae = SparseAutoencoder(
            n_input_features=autoencoder_input_dim,
            n_learned_features=n_learned_features,
            n_components=len(self.args.hook_points)
        ).to(self.device)
        
        # Use custom checkpoint path if provided, otherwise use default
        if hasattr(self.args, 'sae_checkpoint_path') and self.args.sae_checkpoint_path:
            ckpt_path = self.args.sae_checkpoint_path
        else:
            ckpt_path = osp.join(self.args.save_dir_sae_ckpts[self.args.modality], 'sparse_autoencoder_final.pt')
        
        print(f"Loading SAE from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=self.device)
        sae.load_state_dict(state_dict)
        return sae
    
    def extract_batch(self, images):
        """Extract CLIP features and SAE activations for a batch"""
        with torch.no_grad():
            images = images.to(self.device)
            
            # Get CLIP features
            clip_features = self.model.encode_image(images).detach().cpu()
            
            # Get SAE activations from CLIP features
            clip_features_device = clip_features.to(self.device)
            sae_result = self.sae(clip_features_device)
            sae_activations = sae_result.learned_activations.detach().cpu()
            sae_reconstructions = sae_result.decoded_activations.detach().cpu()
        
        return clip_features, sae_activations, sae_reconstructions
    
    def process_dataset(self, dataset, batch_size=32, save_dir=None):
        """Process dataset and extract features"""
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        all_clip_features = []
        all_sae_activations = []
        all_sae_reconstructions = []
        all_indices = []
        
        num_images = len(dataset)
        print(f"\n→ Processing {num_images} images...")
        print(f"  Batch size: {batch_size}")
        print(f"  Expected batches: {(num_images + batch_size - 1) // batch_size}\n")
        
        with tqdm(total=num_images, desc="Extracting features", unit="img") as pbar:
            for batch_idx, (images, indices) in enumerate(dataloader):
                if images is None:
                    continue
                
                # Extract features
                clip_feats, sae_acts, sae_recon = self.extract_batch(images)
                
                all_clip_features.append(clip_feats)
                all_sae_activations.append(sae_acts)
                all_sae_reconstructions.append(sae_recon)
                all_indices.append(indices)
                
                pbar.update(images.shape[0])
        
        # Concatenate all batches
        print("\n→ Concatenating results...")
        clip_features = torch.cat(all_clip_features, dim=0)
        sae_activations = torch.cat(all_sae_activations, dim=0)
        sae_reconstructions = torch.cat(all_sae_reconstructions, dim=0)
        indices = torch.cat(all_indices, dim=0)
        
        results = {
            'clip_features': clip_features,
            'sae_activations': sae_activations,
            'sae_reconstructions': sae_reconstructions,
            'indices': indices,
        }
        
        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"\n→ Saving features to: {save_dir}")
            
            for key, tensor in results.items():
                save_path = osp.join(save_dir, f"{key}.pth")
                torch.save(tensor, save_path)
                print(f"  ✓ {key}: shape {tensor.shape} → {save_path}")
        
        return results


def main():
    """Main extraction pipeline"""
    # Parse arguments
    parser = arg_parser.get_common_parser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="CelebA split to extract"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for extraction"
    )
    parser.add_argument(
        "--celeba_root",
        type=str,
        default="./data/celeba",
        help="Path to CelebA root directory"
    )
    
    parser.add_argument(
    "--sae_checkpoint_path",
    type=str,
    default=None,
    help="Path to SAE checkpoint (if not using default config structure)"
    )

    args = parser.parse_args()
    common_init(args)
    
    # Update config for CelebA (not the default probe datasets)
    args.sae_dataset = "celeba"
    
    # Print setup info
    print("\n" + "="*70)
    print("CLIP + SAE EXTRACTION FOR CELEBA")
    print("="*70)
    print(f"CLIP Model:          {args.img_enc_name}")
    print(f"Device:              {args.device}")
    print(f"CelebA Root:         {args.celeba_root}")
    print(f"Split:               {args.split}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Expansion Factor:    {args.expansion_factor}")
    print(f"SAE Checkpoint:      {osp.join(args.save_dir_sae_ckpts[args.modality], 'sparse_autoencoder_final.pt')}")
    print("="*70)
    
    # Set output directory (using repo's standard structure)
    output_dir = osp.join(
        args.data_dir_root,
        'activations_img',
        'data/activations_img/celeba',
        args.img_enc_name_for_saving,
        args.hook_points[0],
        args.split
    )
    
    print(f"\n→ Output Directory: {output_dir}\n")
    
    # Load CelebA dataset
    try:
        print(f"→ Loading CelebA dataset from: {args.celeba_root}")
        dataset = CelebADataset(
            args.celeba_root,
            preprocess_fn=None,  # Will be set after getting preprocess
            split=args.split
        )
    except Exception as e:
        print(f"\n✗ Error loading CelebA dataset: {e}")
        print("\nExpected directory structure:")
        print("  {celeba_root}/")
        print("  ├── img_align_celeba/     (directory with all 202,599 images)")
        print("  └── list_eval_partition.txt (split assignment file)")
        return False
    
    # Get CLIP preprocessing function
    _, preprocess_fn = get_img_model(args)
    dataset.preprocess = preprocess_fn
    
    # Create extractor and process
    try:
        extractor = CLIPSAEExtractor(args)
        results = extractor.process_dataset(
            dataset,
            batch_size=args.batch_size,
            save_dir=output_dir
        )
    except Exception as e:
        print(f"\n✗ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Print summary
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"CLIP Features:           {results['clip_features'].shape}")
    print(f"SAE Activations:         {results['sae_activations'].shape}")
    print(f"SAE Reconstructions:     {results['sae_reconstructions'].shape}")
    print(f"Total Images Processed:  {len(dataset)}")
    print(f"\n✓ All features saved to: {output_dir}")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)