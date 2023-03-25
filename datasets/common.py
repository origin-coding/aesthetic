from pathlib import Path

import torchvision.transforms as transforms

base_path = (Path(__file__).parent.parent / "data").resolve()  # 数据集所在根目录

batch_size = 100  # 数据集的Batch size

# 处理图像的transform
image_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
])
