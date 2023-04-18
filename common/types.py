from typing import TypedDict, Tuple, Union, Any

from ignite.metrics import Metric
from torch import Tensor


class TensorData(TypedDict):
    binary: Tensor
    attribute: Tensor
    score: Tensor


class TrainData(TypedDict):
    input_tensor: TensorData
    label_tensor: TensorData


TrainStepOutput = Tuple[TensorData, TensorData]


class EngineMetrics(TypedDict):
    loss: Union[Metric, Any]

    # 每个子任务的loss metric和值
    bin_loss: Union[Metric, Any]
    score_loss: Union[Metric, Any]
    attr_loss: Union[Metric, Any]

    # 二分类的准确率
    bin_acc: Union[Metric, Any]
