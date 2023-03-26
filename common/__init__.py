from pathlib import Path

import torchvision.transforms as transforms

base_path = Path(__file__).parent.parent.resolve()  # 项目根目录，即最顶层的aesthetic

batch_size = 100

# 加载图像时对图像做的处理
image_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
])
