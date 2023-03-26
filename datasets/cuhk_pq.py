from torchvision.datasets import ImageFolder

from common import base_path, image_transforms

bin_dataset = ImageFolder(
    root=base_path / "data" / "cuhk_pq",
    transform=image_transforms,
    target_transform=lambda x: 1 - x,  # 这里因为HighQuality在前，所以便签需要进行处理
)
