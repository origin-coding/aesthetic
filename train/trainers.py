from typing import Union

import torch
from ignite.engine import Engine, create_supervised_trainer, Events
from torch.cuda.amp import autocast
from torch.nn import Module

from common.common import TrainData


def setup_evaluator(model: Module, device: Union[str, torch.device]) -> Engine:
    model.to(device)

    @torch.no_grad()
    def eval_function(_: Engine, batch: TrainData):
        model.eval()

        input_tensor = batch.input_tensor
        label_tensor = batch.label_tensor

        with autocast():
            output_tensor = model(input_tensor)

        return output_tensor, label_tensor

    return Engine(eval_function)


if __name__ == '__main__':
    Events.TERMINATE
