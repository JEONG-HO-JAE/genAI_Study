import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Sequence, Union
from torchvision.datasets import CelebA, MNIST
from pytorch_lightning import LightningDataModule

class MyCelebA(CelebA):
    def _check_integrity(self) -> bool:
        return True

    @property
    def base_folder(self):
        return ""


class DDPMDataset(LightningDataModule):
    """
    Combined class for CelebA and MNIST datasets with dynamic transforms.
    """
    def __init__(self,
                 data_path: str,
                 train_batch_size: int = 8,
                 val_batch_size: int = 8,
                 patch_size: Union[int, Sequence[int]] = (256, 256),
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 **kwargs):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Extract dataset name from data_path (e.g., mnist, celeba)
        self.dataset_name = os.path.basename(os.path.normpath(data_path)).lower()

    def prepare_data(self) -> None:
            pass  

    def setup(self, stage: Optional[str] = None) -> None:
        # Define dataset-specific transformations
        if self.dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ])
            if stage == 'fit' or stage is None:
                self.train_dataset = MNIST(self.data_dir, train=True, transform=transform, download=True)
                self.val_dataset = MNIST(self.data_dir, train=False, transform=transform, download=True)
            if stage == 'test' or stage is None:
                self.test_dataset = MNIST(self.data_dir, train=False, transform=transform, download=True)

        elif self.dataset_name == "celeba":
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
            ])
            if stage == 'fit' or stage is None:
                self.train_dataset = MyCelebA(self.data_dir, split='train', transform=transform, download=False)
                self.val_dataset = MyCelebA(self.data_dir, split='valid', transform=transform, download=False)
            if stage == 'test' or stage is None:
                self.test_dataset = MyCelebA(self.data_dir, split='test', transform=transform, download=False)

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )