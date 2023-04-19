from datetime import datetime
from typing import Union, Optional

import torch.cuda
from ignite.engine import Events
from ignite.handlers import BasicTimeProfiler
from ignite.metrics.metric import BatchWise
from torch.cuda.amp import GradScaler
from torch.optim import Adam, SGD

from models import MTAesthetic, MTLoss, MTDwa
from .config import Configuration, OptimizerConfiguration
from .metrics import setup_metrics
from .trainers import setup_trainer, setup_evaluator
from .utils import setup_config, setup_logger, setup_data, log_metrics, setup_checkpoint, setup_exp_logging


def train_main(config_filename: str):
    config: Configuration = setup_config(filename=config_filename)
    # 处理使用AMP加速但是CUDA不可用的情况
    config.use_amp = config.use_amp & torch.cuda.is_available()
    scaler: Optional[GradScaler] = GradScaler() if config.use_amp else None

    logger = setup_logger(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTAesthetic(
        channels=config.channels, kernel_size=config.kernel_size, use_attention=config.use_attention
    ).to(device)

    optimizer: Union[Adam, SGD]
    if config.optimizer == OptimizerConfiguration.ADAM:
        optimizer = Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = SGD(model.parameters(), lr=config.lr)

    loss_fn: Union[MTLoss, MTDwa] = MTLoss() if not config.use_dwa else MTDwa()

    # 创建Engine并添加Metrics、logger和Time Profiler
    train_engine = setup_trainer(model, optimizer, loss_fn, config, device, scaler)
    val_engine = setup_evaluator(model, device, config)
    test_engine = setup_evaluator(model, device, config)

    for engine in (train_engine, val_engine, test_engine):
        engine.logger = logger
        BasicTimeProfiler().attach(engine)

        metrics = setup_metrics(device)
        for key, metric in metrics.items():
            metric.attach(engine, key, usage=BatchWise())  # 每个Batch都记录

    checkpoint = setup_checkpoint(val_engine, {"model": model})

    exp_logger = setup_exp_logging(train_engine, val_engine)

    train_engine.add_event_handler(Events.ITERATION_COMPLETED, log_metrics, tag="train")
    val_engine.add_event_handler(Events.ITERATION_COMPLETED, log_metrics, tag="val")
    test_engine.add_event_handler(Events.ITERATION_COMPLETED, log_metrics, tag="test")

    train_loader, val_loader, test_loader = setup_data(config)

    @train_engine.on(Events.STARTED | Events.EPOCH_COMPLETED)
    def _():
        val_engine.run(val_loader, max_epochs=1)

    @train_engine.on(Events.COMPLETED)
    def _():
        test_engine.run(test_loader, max_epochs=1)

    train_engine.run(train_loader, max_epochs=config.max_epochs)

    exp_logger.close()
    logger.info(f"Finish training at: {datetime.today()}, last checkpoint name: {checkpoint.last_checkpoint}.")
    logger.info("\n")
