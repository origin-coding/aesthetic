from typing import Dict, Tuple

import torch
from ignite.metrics import Metric, Loss, Accuracy, ClassificationReport, MeanSquaredError
from torch import Tensor

from common import TrainStepOutput
from models import MTLoss


def extract_binary(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["binary"].round(), output[1]["binary"]


def extract_score(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["score"], output[1]["score"]


def extract_attribute(output: TrainStepOutput) -> Tuple[Tensor, Tensor]:
    return output[0]["attribute"].round(), output[1]["attribute"]


def setup_metrics(device: torch.device) -> Dict[str, Metric]:
    return {
        "loss": Loss(loss_fn=MTLoss(), device=device),
        "bin": Accuracy(output_transform=extract_binary, device=device),
        "score": MeanSquaredError(output_transform=extract_score, device=device),
        "attribute": ClassificationReport(
            output_dict=True, is_multilabel=True, output_transform=extract_attribute, device=device
        )
    }
