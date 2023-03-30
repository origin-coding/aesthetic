import torch
from torchvision.datasets import ImageFolder

from common import data_path, image_transforms

cuhk_pq_dataset = ImageFolder(
    root=data_path / "cuhk_pq",
    transform=image_transforms,
    target_transform=lambda x: torch.tensor(1 - x),  # 这里因为HighQuality在前，所以便签需要进行处理
)
