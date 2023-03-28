from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import base_path, image_transforms


class AADBDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv(base_path / "data" / "aadb.csv")
        self.transform = image_transforms

    def __len__(self) -> int:
        return 10000  # 这里直接写死成10000，也可以写成下面这种通过数据数量计算的方式
        # return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        items: list = self.data.iloc[index].to_list()

        image = Image.open(base_path / "data" / "aadb" / f"{items[0]}")
        input_tensor = self.transform(image)
        label_tensor = torch.tensor(items[1:])

        return input_tensor, label_tensor
