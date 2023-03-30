from pathlib import Path

from pydantic import BaseModel
from torch import Tensor
from torchvision import transforms as transforms

image_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.RandomCrop(256),
    transforms.ToTensor(),
])


class TensorData(BaseModel):
    binary: Tensor
    attribute: Tensor
    score: Tensor


class TrainData(BaseModel):
    input_tensor: TensorData
    label_tensor: TensorData


base_path = Path(__file__).parent.parent.resolve()  # 项目根目录，即最顶层的aesthetic
data_path = base_path / "data"
output_path = base_path / "outputs"
