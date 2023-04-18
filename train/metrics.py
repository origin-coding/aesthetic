from typing import Tuple

import torch
from ignite.metrics import Loss, Accuracy
from torch import Tensor
from torch.nn import MultiLabelMarginLoss, BCEWithLogitsLoss, MSELoss

from common import TrainStepOutput
from models import MTLoss
from .config import EngineMetrics


def extract_binary(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["binary"].float(), output[1]["binary"].float()


def extract_binary_accuracy(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["binary"].round().float(), output[1]["binary"].float()


def extract_score(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["score"].float(), output[1]["score"].float()


def extract_attribute(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["attribute"].float(), output[1]["attribute"].long()


def setup_metrics(device: torch.device) -> EngineMetrics:
    return {
        "loss": Loss(loss_fn=MTLoss(), device=device),
        "bin_loss": Loss(loss_fn=BCEWithLogitsLoss(), output_transform=extract_binary, device=device),
        "score_loss": Loss(loss_fn=MSELoss(), output_transform=extract_score, device=device),
        "attr_loss": Loss(loss_fn=MultiLabelMarginLoss(), output_transform=extract_attribute, device=device),
        "bin_acc": Accuracy(output_transform=extract_binary_accuracy, device=device),
    }
