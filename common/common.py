from pathlib import Path

from PIL import Image
from torchvision import transforms as transforms


def convert_to_rgb(image: Image) -> Image:
    return image.convert("RGB")


# 图像预处理，尽量不改变图像的颜色、亮度、方位等特征，最大程度保留图像的美学信息
image_transforms = transforms.Compose([
    convert_to_rgb,
    transforms.Resize(512),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
])

base_path = Path(__file__).parent.parent.resolve()  # 项目根目录，即最顶层的aesthetic
data_path = base_path / "data"
output_path = base_path / "outputs"
checkpoint_path = output_path / "checkpoints"
log_path = output_path / "logs"
