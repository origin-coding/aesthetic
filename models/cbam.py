import torch
import torch.nn as nn


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
    """
    CBAM：Convolutional Block Attention Module，https://arxiv.org/pdf/1807.06521
    在图像处理中使用Attention机制，上面的ChannelAttention和SpecialAttention即是CBAM的两个子模块
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
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
