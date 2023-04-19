from enum import Enum
from typing import TypedDict, Union, Any

from ignite.metrics import Metric
from pydantic import BaseModel


class OptimizerConfiguration(str, Enum):
    ADAM = "adam"
    SGD = "sgd"


class Configuration(BaseModel):
    batch_size: int = 100
    lr: float = 1e-4

    max_epochs: int = 30

    use_amp: bool  # 是否使用混合精度计算加速模型训练

    optimizer: OptimizerConfiguration = OptimizerConfiguration.ADAM
    channels: int = 1024  # Attention模块的通道数，建议取值为1024或512
    kernel_size: int = 3  # SharedLayer的卷积核大小，建议取值为3、5、7

    use_attention: bool = True  # 是否使用Attention，用于模型对比
    use_dwa: bool = False  # 是否使用DWA平衡loss，用于损失函数对比


class EngineMetrics(TypedDict):
    loss: Union[Metric, Any]

    # 每个子任务的loss metric和值
    bin_loss: Union[Metric, Any]
    score_loss: Union[Metric, Any]
    attr_loss: Union[Metric, Any]

    # 二分类的准确率
    bin_acc: Union[Metric, Any]
