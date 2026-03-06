import os
import torch
from tqdm.auto import tqdm
from pprint import pprint
import os.path as osp
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from PIL import Image

# From custom
from dncbm.utils import common_init, get_img_model
from dncbm import arg_parser


class CelebADataset(Dataset):
    """
    CelebA dataset loader compatible with CLIP preprocessing.

    Expected directory structure:
        {celeba_root}/
        ├── img_align_celeba/          (202,599 aligned face images)
        └── list_eval_partition.txt    (per-image split: 0=train, 1=val, 2=test)
    """

    def __init__(self, root_dir, preprocess_fn, split='train'):
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / 'img_align_celeba'
        self.preprocess = preprocess_fn
        self.split = split

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        split_file = self.root_dir / 'list_eval_partition.txt'
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        split_map = {'train': 0, 'val': 1, 'test': 2}
        partition_idx = split_map[split]

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

        print(f"Loaded {len(self.image_files)} images for '{split}' split from CelebA")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = self.img_dir / img_name
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img)
            return img_tensor, idx
        except Exception as e:
            print(f"Warning: error loading {img_path}: {e}")
            return None, idx


def collate_fn(batch):
    """Skip any None images from failed loads."""
    batch = [(img, idx) for img, idx in batch if img is not None]
    if not batch:
        return None, None
    imgs, idxs = zip(*batch)
    return torch.stack(imgs), list(idxs)


class FetchFeatures:
    def __init__(self, args):
        self.model, self.preprocess = get_img_model(args)
        self.args = args

    def get_celeba_loader(self, split, batch_size):
        dataset = CelebADataset(
            root_dir=self.args.celeba_root,
            preprocess_fn=self.preprocess,
            split=split,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )
        return loader

    def get_features(self, loader):
        count = 0
        out = None
        with torch.no_grad():
            for images, idxs in tqdm(loader, desc="Extracting CLIP features", unit="batch"):
                if images is None:
                    continue
                count += images.shape[0]
                images = images.to(self.args.device)
                feats = self.model.encode_image(images).detach().cpu()
                out = feats if out is None else torch.vstack((out, feats))
                print(f"  total processed: {count}")
        return out

    def save_celeba_features(self):
        save_dir = self.args.data_dir_activations[self.args.modality]

        # ── FIX: ensure the output directory exists before writing into it ──
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving features to: {save_dir}")

        # Step 1: save val and test splits directly
        for split in ['val', 'test']:
            print(f"\n--- Processing '{split}' split ---")
            loader = self.get_celeba_loader(split, batch_size=self.args.batch_size)
            features = self.get_features(loader)
            save_path = osp.join(save_dir, split)
            torch.save(features, save_path)
            print(f"Saved {split} features {features.shape} -> {save_path}")

        # Step 2: load train split and split 90/10 into train + train_val
        # (train_val is used by the SAE pipeline for validation during training)
        print(f"\n--- Processing 'train' split ---")
        loader = self.get_celeba_loader('train', batch_size=self.args.batch_size)
        train_feats = self.get_features(loader)
        print(f"  Full train features: {train_feats.shape}")

        n_total = train_feats.shape[0]
        n_val = int(0.1 * n_total)
        perm = torch.randperm(n_total)

        train_feats_sub  = train_feats[perm[n_val:]]   # 90% → train
        train_val_feats  = train_feats[perm[:n_val]]   # 10% → train_val

        torch.save(train_feats_sub, osp.join(save_dir, 'train'))
        torch.save(train_val_feats, osp.join(save_dir, 'train_val'))
        print(f"  Saved train     {train_feats_sub.shape} -> {osp.join(save_dir, 'train')}")
        print(f"  Saved train_val {train_val_feats.shape} -> {osp.join(save_dir, 'train_val')}")

        print(f"\nAll CelebA features saved to: {save_dir}")


if __name__ == '__main__':
    parser = arg_parser.get_common_parser()
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for CLIP feature extraction")
    parser.add_argument("--celeba_root", type=str, default="./data/celeba",
                        help="Path to CelebA root directory")
    args = parser.parse_args()

    args.sae_dataset = "celeba"
    common_init(args)
    pprint(vars(args))

    fetch = FetchFeatures(args)
    fetch.save_celeba_features()