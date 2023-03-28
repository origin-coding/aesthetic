from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset

from .aadb import AADBDataset
from .ava import AVADataset
from .cuhk_pq import cuhk_pq_dataset

__all__ = ["cuhk_pq_dataset", "AADBDataset", "AVADataset"]


class MTAestheticDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.cuhk_pq = cuhk_pq_dataset
        self.aadb = AADBDataset()
        self.ava = AVADataset()

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, index) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        return self.ava[index], self.aadb[index], self.cuhk_pq[index]
