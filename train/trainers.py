from typing import Union, Optional, Tuple

import torch
from ignite.engine import Engine, DeterministicEngine
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, SGD

from common import TrainData, TensorData
from models import MTAesthetic, MTLoss
from .config import Configuration


def prepare_batch(batch: TrainData, device: torch.device, non_blocking: bool = True) -> TrainData:
    input_tensor = batch.input_tensor
    label_tensor = batch.label_tensor

    input_binary = input_tensor.binary.to(device, non_blocking=non_blocking)
    input_score = input_tensor.score.to(device, non_blocking=non_blocking)
    input_attribute = input_tensor.attribute.to(device, non_blocking=non_blocking)

    label_binary = label_tensor.binary.to(device, non_blocking=non_blocking)
    label_score = label_tensor.score.to(device, non_blocking=non_blocking)
    label_attribute = label_tensor.attribute.to(device, non_blocking=non_blocking)

    return TrainData(
        input_tensor=TensorData(binary=input_binary, score=input_score, attribute=input_attribute),
        label_tensor=TensorData(binary=label_binary, score=label_score, attribute=label_attribute)
    )


def setup_trainer(
        model: MTAesthetic, optimizer: Union[Adam, SGD], loss_fn: MTLoss, config: Configuration,
        device: torch.device = torch.device("cuda"), scaler: Optional[GradScaler] = None
) -> Union[Engine, DeterministicEngine]:
    if config.use_amp:
        assert scaler is not None

    def train_step(_: Engine, batch: TrainData) -> Tuple[TensorData, TensorData]:
        # 将模型更改至训练模式
        optimizer.zero_grad()
        model.train()

        batch = prepare_batch(batch, device)
        input_tensor: TensorData = batch.input_tensor
        label_tensor: TensorData = batch.label_tensor

        with autocast(config.use_amp):
            # 计算输出结果、损失值
            output_tensor: TensorData = model(input_tensor)
            loss: torch.Tensor = loss_fn(output_tensor, label_tensor)

        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        return output_tensor, label_tensor

    trainer = Engine(train_step)

    return trainer


def setup_evaluator(model: MTAesthetic, device: torch.device, config: Configuration) -> Engine:
    def evaluate_step(_: Engine, batch: TrainData) -> Tuple[TensorData, TensorData]:
        model.eval()

        with torch.no_grad():
            batch = prepare_batch(batch, device)
            input_tensor: TensorData = batch.input_tensor
            label_tensor: TensorData = batch.label_tensor

            with autocast(config.use_amp):
                output_tensor: TensorData = model(input_tensor)

            return output_tensor, label_tensor

    return Engine(evaluate_step)
