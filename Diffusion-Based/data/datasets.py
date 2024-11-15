from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Sequence, Union
from torchvision.datasets import CelebA
from pytorch_lightning import LightningDataModule

class MyCelebA(CelebA):
    def _check_integrity(self) -> bool:
        return True

    @property
    def base_folder(self):
        # 필요한 데이터가 있는 폴더로 base_folder를 설정
        return ""

class DDPMDataset(LightningDataModule):
    """
    Combined class for CelebA dataset with custom integrity check and LightningDataModule.
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

    def prepare_data(self) -> None:
        # 다운로드를 비활성화하므로 추가 작업 불필요
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        
        # Define transformations
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(148),
                transforms.Resize(self.patch_size),
                transforms.ToTensor(),
        ])
        

        # Dataset splits
        if stage == 'fit' or stage is None:
            self.train_dataset = MyCelebA(
                self.data_dir,
                split='train',
                transform=transform,
                download=False  # 다운로드 비활성화
            )
            self.val_dataset = MyCelebA(
                self.data_dir,
                split='valid',
                transform=transform,
                download=False  # 다운로드 비활성화
            )
        if stage == 'test' or stage is None:
            self.test_dataset = MyCelebA(
                self.data_dir,
                split='test',
                transform=transform,
                download=False  # 다운로드 비활성화
            )

    def _check_integrity(self) -> bool:
        # Override to skip the integrity check
        return True

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