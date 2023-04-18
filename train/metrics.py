from typing import Tuple

import torch
from ignite.metrics import Loss, Accuracy, MeanSquaredError
from torch import Tensor
from torch.nn import MultiLabelMarginLoss, BCEWithLogitsLoss

from common import TrainStepOutput, EngineMetrics
from models import MTLoss


def extract_binary(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["binary"].round(), output[1]["binary"]


def extract_score(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["score"], output[1]["score"]


def extract_attribute(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["attribute"].round(), output[1]["attribute"]


def setup_metrics(device: torch.device) -> EngineMetrics:
    return {
        "loss": Loss(loss_fn=MTLoss(), device=device),
        "bin_loss": Loss(loss_fn=BCEWithLogitsLoss(), output_transform=extract_binary, device=device),
        "score_loss": MeanSquaredError(output_transform=extract_score, device=device),
        "attr_loss": Loss(loss_fn=MultiLabelMarginLoss(), output_transform=extract_attribute, device=device),
        "bin_acc": Accuracy(output_transform=extract_binary, device=device),
    }
