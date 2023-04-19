import torch
import torch.nn as nn

from common import TensorData
from .cbam import CBAM


class SharedLayer(nn.Module):
    """多任务学习的共享参数层"""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()

        self.feature1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=128),
        )

        self.feature2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=64),
        )

        self.feature3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=32),
        )

        self.feature4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=16),
        )

        self.feature5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=1024, out_channels=channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=8),
        )

        self.layers = nn.Sequential(
            self.feature1,
            self.feature2,
            self.feature3,
            self.feature4,
            self.feature5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return out


class MTAesthetic(nn.Module):
    def __init__(self, channels: int, kernel_size: int, use_attention: bool = True) -> None:
        super().__init__()

        # 共享参数层，用于学习图像的细节特征
        self.shared_layer = SharedLayer(channels=channels, kernel_size=kernel_size)

        # 三个多任务学习模块
        self.task_binary = nn.Sequential(
            CBAM(channels=channels) if use_attention else nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(), nn.ReLU(), nn.Dropout(),
            nn.Linear(in_features=channels * 1 * 1, out_features=1),
            nn.Sigmoid()
        )

        self.task_score = nn.Sequential(
            CBAM(channels=channels) if use_attention else nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(), nn.ReLU(), nn.Dropout(),
            nn.Linear(in_features=channels * 1 * 1, out_features=10),
            nn.Softmax(dim=1)
        )

        self.task_attribute = nn.Sequential(
            CBAM(channels=channels) if use_attention else nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(), nn.ReLU(), nn.Dropout(),
            nn.Linear(in_features=channels * 1 * 1, out_features=11),
            nn.Sigmoid()
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)

            if isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight.data)
                nn.init.zeros_(module.bias.data)

            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, mean=0, std=0.01)
                nn.init.zeros_(module.bias.data)

    def forward(self, input_tensors: TensorData) -> TensorData:
        # 首先获取输入
        input_binary = input_tensors["binary"]
        input_score = input_tensors["score"]
        input_attribute = input_tensors["attribute"]

        # 针对三个子任务分别进行输出
        output_binary = self.shared_layer(input_binary)
        output_binary = self.task_binary(output_binary)
        output_binary = output_binary.squeeze()

        output_score = self.shared_layer(input_score)
        output_score = self.task_score(output_score)

        output_attribute = self.shared_layer(input_attribute)
        output_attribute = self.task_attribute(output_attribute)

        return TensorData(binary=output_binary, score=output_score, attribute=output_attribute)
