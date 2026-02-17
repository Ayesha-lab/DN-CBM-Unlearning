import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class BinaryMNISTLoader:
    def __init__(self, data_dir='./Datasets', batch_size=32):
        """
        Load MNIST dataset for binary classification (even vs odd)
        and create custom retain/forget splits.

        Args:
            data_dir: directory to download/load MNIST data
            batch_size: batch size for DataLoader
        """
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define transforms for MNIST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])

    def _create_binary_target(self, target):
        """Convert digit labels to binary (0=even, 1=odd)"""
        return target % 2

    def load_data(self):
        """
        Load MNIST and create train/test/retain/forget splits.

        Binary classification: even (0) vs odd (1)
        Forget set: all images of digit 1
        Retain set: all images of digits 0, 2, 3, 4, 5, 6, 7, 8, 9
        """
        # Load datasets
        full_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )

        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )

        # Create a wrapper to handle binary labels
        train_dataset_binary = BinaryMNISTDataset(full_dataset)
        test_dataset_binary = BinaryMNISTDataset(test_dataset)

        # Split training data into retain (digits 0,2-9) and forget (digit 1)
        retain_indices = []
        forget_indices = []

        for idx, (_, label) in enumerate(full_dataset):
            if label == 1:  # digit 1 goes to forget set
                forget_indices.append(idx)
            else:  # digits 0, 2-9 go to retain set
                retain_indices.append(idx)

        retain_dataset = BinaryMNISTDataset(Subset(full_dataset, retain_indices))
        forget_dataset = BinaryMNISTDataset(Subset(full_dataset, forget_indices))

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset_binary,
            batch_size=self.batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset_binary,
            batch_size=self.batch_size,
            shuffle=False
        )

        retain_loader = DataLoader(
            retain_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        forget_loader = DataLoader(
            forget_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        print(f"Dataset sizes:")
        print(f"  Train (all): {len(train_dataset_binary)}")
        print(f"  Retain (digits 0,2-9): {len(retain_dataset)}")
        print(f"  Forget (digit 1): {len(forget_dataset)}")
        print(f"  Test: {len(test_dataset_binary)}")

        return train_loader, test_loader, retain_loader, forget_loader


class BinaryMNISTDataset:
    """Wrapper to convert MNIST labels to binary (even=0, odd=1)"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(self.dataset, Subset):
            img, label = self.dataset[idx]
        else:
            img, label = self.dataset[idx]

        # Convert to binary: even=0, odd=1
        binary_label = label % 2
        return img, binary_label