from enum import Enum

from pydantic import BaseModel

from .common import base_path


class OptimizerConfiguration(str, Enum):
    ADA = "ada"


class Configuration(BaseModel):
    batch_size: int = 100
    max_epochs: int = 200
    lr: float = 1e-4

    use_amp: bool  # 是否使用混合精度计算加速模型训练

    optimizer: OptimizerConfiguration = OptimizerConfiguration.ADA

    channels: int = 1024
    shared_layer_kernel_size: int = 3


def setup_config() -> Configuration:
    return Configuration.parse_file(path=base_path / "config.json", content_type="json")
