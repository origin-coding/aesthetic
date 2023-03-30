from enum import Enum

from pydantic import BaseModel

from .common import base_path


class OptimizerConfiguration(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class Configuration(BaseModel):
    batch_size: int = 100
    lr: float = 1e-4

    max_epochs: int = 200
    patience: int = 5  # 如果连续多个epoch结果不再变好，那么停止训练

    use_amp: bool  # 是否使用混合精度计算加速模型训练

    optimizer: OptimizerConfiguration = OptimizerConfiguration.ADAM
    channels: int = 1024  # Attention模块的通道数，建议取值为1024或512
    kernel_size: int = 3  # SharedLayer的卷积核大小，建议取值为3、5、7


def setup_config() -> Configuration:
    return Configuration.parse_file(path=base_path / "config.json", content_type="json")
