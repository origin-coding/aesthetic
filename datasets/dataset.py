from torch.utils.data import Dataset

from common import TrainData, TensorData
from .aadb import AADBDataset
from .ava import AVADataset
from .cuhk_pq import cuhk_pq_dataset


class MTAestheticDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.cuhk_pq = cuhk_pq_dataset
        self.aadb = AADBDataset()
        self.ava = AVADataset()

        assert len(self.cuhk_pq) == 10000
        assert len(self.ava) == 10000
        assert len(self.aadb) == 10000

    def __len__(self) -> int:
        return 10000

    def __getitem__(self, index: int) -> TrainData:
        train_data = TrainData(
            input_tensor=TensorData(
                binary=self.cuhk_pq[index][0], score=self.ava[index][0], attribute=self.aadb[index][0]
            ),
            label_tensor=TensorData(
                binary=self.cuhk_pq[index][1], score=self.ava[index][1], attribute=self.aadb[index][1]
            )
        )
        return train_data
