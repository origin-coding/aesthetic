from typing import Union, Optional

import torch
from ignite.engine import Engine
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, SGD

from common import TrainData, TensorData, StepOutput
from models import MTAesthetic, MTLoss, MTDwa
from .config import Configuration


def prepare_batch(batch: TrainData, device: torch.device, non_blocking: bool = True) -> TrainData:
    """
    用于准备每个batch用到的数据，把数据从CPU迁移到GPU上
    :param batch: 一个batch的数据
    :param device: 需要迁移的device，主要是GPU
    :param non_blocking: 是否进行non_blocking迁移
    :return: 准备好的数据
    """
    input_tensor = batch["input_tensor"]
    label_tensor = batch["label_tensor"]

    input_binary = input_tensor["binary"].to(device, non_blocking=non_blocking)
    input_score = input_tensor["score"].to(device, non_blocking=non_blocking)
    input_attribute = input_tensor["attribute"].to(device, non_blocking=non_blocking)

    label_binary = label_tensor["binary"].to(device, non_blocking=non_blocking)
    label_score = label_tensor["score"].to(device, non_blocking=non_blocking)
    label_attribute = label_tensor["attribute"].to(device, non_blocking=non_blocking)

    return TrainData(
        input_tensor=TensorData(binary=input_binary, score=input_score, attribute=input_attribute),
        label_tensor=TensorData(binary=label_binary, score=label_score, attribute=label_attribute)
    )


def setup_trainer(
        model: MTAesthetic, optimizer: Union[Adam, SGD], loss_fn: Union[MTLoss, MTDwa], config: Configuration,
        device: torch.device = torch.device("cuda"), scaler: Optional[GradScaler] = None
) -> Engine:
    """
    用于创建训练过程的engine
    :param model: 训练模型用到的Model
    :param optimizer: 训练模型的optimizer
    :param loss_fn: 损失函数，主要是MTLoss和MTDwa两种
    :param config: 配置项，用到use_amp配置项
    :param device: 数据存放的device，主要是GPU
    :param scaler: 使用AMP混合精度计算时的scaler，用于加速模型训练
    :return: train engine
    """
    if config.use_amp:
        assert scaler is not None

    def train_step(_: Engine, batch: TrainData) -> StepOutput:
        """
        训练过程的每个step: 1. 将模型调制训练模式, 2. 准备数据，用模型进行计算，获取输出, 3. 计算损失函数、梯度下降、反向传播, 4. 返回数据
        :param _: 训练用的engine，ignite要求step函数的第一个参数是engine
        :param batch: 每个batch的数据
        :return: 每个step的输出，包括output_tensor(y)和label_tensor(y_pred)
        """

        # 将模型更改至训练模式
        optimizer.zero_grad()
        model.train()

        # 将数据迁移至对应的设备上
        batch = prepare_batch(batch, device)
        input_tensor: TensorData = batch["input_tensor"]
        label_tensor: TensorData = batch["label_tensor"]

        with autocast(config.use_amp):
            # 计算输出结果、损失值
            output_tensor: TensorData = model(input_tensor)
            loss: torch.Tensor = loss_fn(output_tensor, label_tensor)

        # 梯度下降、反向传播
        if config.use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 返回数据
        return output_tensor, label_tensor

    trainer = Engine(train_step)
    return trainer


def setup_evaluator(model: MTAesthetic, device: torch.device, config: Configuration) -> Engine:
    """
    用于创建验证/测试过程的engine，主要是val engine和test engine
    :param model: 用于验证的模型
    :param device: 数据存放的device，主要是GPU
    :param config: 配置项，用到use_amp
    :return: 创建完毕的engine
    """
    def evaluate_step(_: Engine, batch: TrainData) -> StepOutput:
        """
        验证/测试过程的step: 1. 将模型转换到验证模式, 2. 准备数据，用模型进行计算，获取输出, 3. 返回数据
        :param _: 用于验证的engine，ignite要求step函数的第一个参数是engine
        :param batch: 每个batch的数据
        :return: 每个step的输出，包括output_tensor(y)和label_tensor(y_pred)
        """
        # 将模型调整至验证模式
        model.eval()

        with torch.no_grad():
            # 准备数据
            batch = prepare_batch(batch, device)
            input_tensor: TensorData = batch["input_tensor"]
            label_tensor: TensorData = batch["label_tensor"]

            # 获取输出并返回结果
            with autocast(config.use_amp):
                output_tensor: TensorData = model(input_tensor)
            return output_tensor, label_tensor

    return Engine(evaluate_step)
