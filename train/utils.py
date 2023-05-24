from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import loguru
import torch
from ignite.contrib.engines.common import setup_tb_logging, TensorboardLogger
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver, Checkpoint, global_step_from_engine
from torch.utils.data import random_split, DataLoader

from common import base_path, log_path, checkpoint_path
from datasets import MTAestheticDataset
from .config import Configuration, EngineMetrics


def setup_config(filename: str = "config.json") -> Configuration:
    """
    用于加载配置文件的内容
    :param filename: 配置文件的文件名，默认放在根目录下，为config.json
    :return: 加载好的配置项
    """
    return Configuration.parse_file(path=base_path / filename, content_type="json")


def setup_data(config: Configuration) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    用于创建、分割、打乱数据集以及创建DataLoader
    :param config: 加载数据集用到的配置项，用到了batch_size
    :return: 返回一个元组，分别是训练、验证和测试用到的DataLoader
    """
    dataset = MTAestheticDataset()
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[6000, 2000, 2000])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    return train_loader, val_loader, test_loader


def setup_logger(config: Configuration, test: bool = False) -> "loguru.Logger":
    """
    用于加载训练时记录信息的logger，包括基本的参数，训练中的数据、花费时间等，并记录下初始的日志
    :param config: 配置项，用到了绝大多数配置项
    :param test: 是否在测试环境中，如果是，那么关闭控制台输出
    :return: 配置好的logger
    """
    logger = loguru.logger
    if test:
        logger.remove(handler_id=None)

    logger.add(log_path / f"{datetime.today().date()}.log", level="INFO")

    logger.info(f"Begin training at: {datetime.today()}.")
    logger.info(
        f"Model info: optimizer: {config.optimizer}, channels: {config.channels}, "
        f"kernel size: {config.kernel_size}, learning rate: {config.lr}."
    )
    logger.info(f"Training info: batch size: {config.batch_size}, epoch counts: {config.max_epochs}.")
    logger.info(f"Using attention(CBAM): {config.use_attention}, using DWA: {config.use_dwa}.")
    logger.info(f"Using amp: {config.use_amp}.")

    return loguru.logger


def log_metrics(engine: Engine, tag: str) -> None:
    """
    用于在训练和验证时保存训练的指标
    :param engine: ignite的engine，主要是train engine和val engine
    :param tag: 保存日志的tag，分别是train和val
    :return: None
    """

    logger = engine.logger

    state = engine.state
    metrics: EngineMetrics = state.metrics

    logger.info(f"Stage: {tag}, epoch: {state.epoch}, iteration: {state.iteration}.")
    logger.info(
        f"Loss: {metrics['loss']:.3f}, bin_loss: {metrics['bin_loss']:.6f}, "
        f"score_loss: {metrics['score_loss']:.6f}, attr_loss: {metrics['attr_loss']:.6f}, "
        f"bin_acc: {metrics['bin_acc']:.3f}."
    )


def setup_checkpoint(engine: Engine, to_save: dict) -> Checkpoint:
    """
    用于保存验证过程的模型
    :param engine: 用于附加checkpoint的engine
    :param to_save: 需要保存的内容，可能是模型权重，也可以带上optimizer等其他内容
    :return: 附加好的checkpoint handler
    """
    saver = DiskSaver(dirname=checkpoint_path, require_empty=False)

    checkpoint = Checkpoint(
        to_save=to_save,
        save_handler=saver, n_saved=3,
        filename_prefix="best",
        score_name="loss", score_function=Checkpoint.get_default_score_fn("loss", score_sign=-1.0),
        global_step_transform=global_step_from_engine(engine)
    )

    engine.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpoint)
    return checkpoint


def resume_from(to_load: dict, checkpoint_fp: Union[str, Path], strict: bool = True) -> None:
    """
    用于从Checkpoint中加载模型等对象
    :param to_load: 需要加载的对象
    :param checkpoint_fp: checkpoint所在路径
    :param strict: 是否使用strict模式
    :return: None
    """
    if isinstance(checkpoint_fp, str):
        checkpoint_fp = Path(checkpoint_fp)
    if not checkpoint_fp.exists():
        raise FileNotFoundError(f"Given {str(checkpoint_fp)} does not exist.")

    checkpoint = torch.load(checkpoint_fp, map_location="cpu")
    Checkpoint.load_objects(to_load, checkpoint=checkpoint, strict=strict)


def setup_exp_logging(trainer: Engine, evaluator: Engine) -> TensorboardLogger:
    """
    用于创建TensorBoard(X)的日志内容，用于保存训练过程中的metrics等内容
    :param trainer: 需要保存数据的train engine
    :param evaluator: 需要保存数据的val engine
    :return: 配置好的logger
    """
    # 每10个iteration保存一次
    logger = setup_tb_logging(output_path=log_path, trainer=trainer, evaluators=evaluator, log_every_iters=10)
    return logger
