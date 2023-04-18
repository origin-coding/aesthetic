from datetime import datetime
from typing import Tuple

import loguru
from ignite.contrib.engines.common import setup_tb_logging, TensorboardLogger
from ignite.engine import Engine, Events
from ignite.handlers import DiskSaver, Checkpoint, global_step_from_engine
from torch.optim import Optimizer
from torch.utils.data import random_split, DataLoader

from common import base_path, log_path, checkpoint_path
from datasets import MTAestheticDataset
from .config import Configuration, EngineMetrics


def setup_config(filename: str = "config.json") -> Configuration:
    return Configuration.parse_file(path=base_path / filename, content_type="json")


def setup_data(config: Configuration) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = MTAestheticDataset()
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[6000, 2000, 2000])

    # 这里在本地验证时为节省资源，而不设置num_workers
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=20)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=20)

    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def setup_logger(config: Configuration) -> "loguru.Logger":
    logger = loguru.logger

    logger.add(log_path / f"{datetime.today().date()}.log", level="INFO")

    logger.info(f"Begin training at: {datetime.today()}.")
    logger.info(
        f"Model info: optimizer: {config.optimizer}, channels: {config.channels}, "
        f"kernel size: {config.kernel_size}, learning rate: {config.lr}."
    )
    logger.info(f"Training info: batch size: {config.batch_size}, epoch counts: {config.max_epochs}.")
    logger.info(f"Using amp: {config.use_amp}.")

    return loguru.logger


def log_metrics(engine: Engine, tag: str) -> None:
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


def setup_exp_logging(trainer: Engine, evaluator: Engine, optimizer: Optimizer) -> TensorboardLogger:
    logger = setup_tb_logging(
        output_path=log_path, trainer=trainer, evaluators=evaluator, optimizers=optimizer, log_every_iters=1
    )
    return logger
