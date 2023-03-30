import torch
import torch.nn as nn

from common import TensorData
from .cbam import CBAM


class SharedLayer(nn.Module):
    """多任务学习的共享参数层"""

    def __init__(self, output_channels: int = 1024, output_size: int = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.feature1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(output_size=128),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU()
        )

        self.feature2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(output_size=output_size),
            nn.Conv2d(in_channels=3, out_channels=output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=output_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature1(x)
        out = self.feature2(out)
        return out


class MTAesthetic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # 共享参数层，用于学习图像的细节特征
        self.shared_layer = SharedLayer()

        # 三个多任务学习模块
        self.task_binary = nn.Sequential(
            CBAM(),
            nn.Flatten(),
            nn.Linear(in_features=1024 * 16 * 16, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1)
        )

        self.task_score = nn.Sequential(
            CBAM(),
            nn.Flatten(),
            nn.Linear(in_features=1024 * 16 * 16, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=1),
        )

        self.task_attribute = nn.Sequential(
            CBAM(),
            nn.Flatten(),
            nn.Linear(in_features=1024 * 16 * 16, out_features=1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=1024, out_features=11)
        )

    def forward(self, input_tensors: TensorData) -> TensorData:
        # 首先获取输入
        input_binary = input_tensors.binary
        input_score = input_tensors.score
        input_attribute = input_tensors.attribute

        # 针对三个子任务分别进行输出
        output_binary = self.shared_layer(input_binary)
        output_binary = self.task_binary(output_binary)

        output_score = self.shared_layer(input_score)
        output_score = self.task_score(output_score)

        output_attribute = self.shared_layer(input_attribute)
        output_attribute = self.task_attribute(output_attribute)

        return TensorData(binary=output_binary, score=output_score, attribute=output_attribute)
