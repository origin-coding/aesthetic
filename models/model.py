import torch
import torch.nn as nn

from common import Task


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduce_ratio: int = 16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(None, None))
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=(None, None))

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channels, channels // reduce_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduce_ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpacialAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat((avg_out, max_out), dim=1)

        return self.sigmoid(self.conv2d(out))


class CBAM(nn.Module):
    def __init__(self, channels: int = 1024, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channel_attention = ChannelAttention(channels)
        self.spacial_attention = SpacialAttention()
        self.activation = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_attention(x) * x
        out = self.spacial_attention(out) * out
        out = self.activation(out)
        return out


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

    def forward(self, x: torch.Tensor, task: Task) -> torch.Tensor:
        out = self.shared_layer(x)

        if task == Task.BINARY:
            out = self.task_binary(out)
        elif task == Task.SCORE:
            out = self.task_score(out)
        elif task == Task.ATTRIBUTE:
            out = self.task_attribute(out)

        return out
