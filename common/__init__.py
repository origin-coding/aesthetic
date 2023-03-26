from enum import Enum
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


# 图像美学评价的任务
class Task(Enum):
    BINARY = "bin"  # 图像是否具有美学特征的二分类任务
    SCORE = "score"  # 图像美学评价的具体分值
    ATTRIBUTE = "attr"  # 图像美学具体特征的指标
