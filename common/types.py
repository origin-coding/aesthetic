from typing import TypedDict, Tuple

from torch import Tensor


class TensorData(TypedDict):
    binary: Tensor
    attribute: Tensor
    score: Tensor


class TrainData(TypedDict):
    input_tensor: TensorData
    label_tensor: TensorData


TrainStepOutput = Tuple[TensorData, TensorData]
