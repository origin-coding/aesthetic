from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import data_path, image_transforms


class AVADataset(Dataset):
    """
    处理后的AVA数据集
    """
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv(data_path / "ava.csv", dtype={"img_id": str, "score": float})
        self.transform = image_transforms

    def __len__(self) -> int:
        return 10000  # 这里直接写死成10000，也可以写成下面这种通过数据数量计算的方式
        # return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        items: list = self.data.iloc[index].to_list()

        image = Image.open(data_path / "ava" / f"{items[1]}.jpg")
        input_tensor = self.transform(image)

        # 将打分人数转换为概率分布
        scores = []
        count = sum(items[2:12])
        for i in items[2:12]:
            scores.append(i / count)

        label_tensor = torch.tensor(scores, dtype=torch.float)

        return input_tensor, label_tensor
