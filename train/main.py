from datetime import datetime
from typing import Union, Optional

import torch.cuda
from ignite.engine import Events
from torch.cuda.amp import GradScaler
from torch.optim import Adam, SGD

from models import MTAesthetic, MTLoss
from .config import Configuration, OptimizerConfiguration
from .trainers import setup_trainer, setup_evaluator
from .utils import setup_config, setup_logger, setup_data


def train_main(config_filename: str):
    config: Configuration = setup_config(filename=config_filename)
    # 处理使用AMP加速但是CUDA不可用的情况
    config.use_amp = config.use_amp & torch.cuda.is_available()

    logger = setup_logger(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTAesthetic(channels=config.channels, kernel_size=config.kernel_size).to(device)

    optimizer: Union[Adam, SGD]
    if config.optimizer == OptimizerConfiguration.ADAM:
        optimizer = Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = SGD(model.parameters(), lr=config.lr)

    loss_fn = MTLoss()
    scaler: Optional[GradScaler] = GradScaler() if config.use_amp else None

    train_engine = setup_trainer(model, optimizer, loss_fn, config, device, scaler)
    val_engine = setup_evaluator(model, device, config)
    test_engine = setup_evaluator(model, device, config)

    train_loader, val_loader, test_loader = setup_data(config)

    @train_engine.on(Events.STARTED)
    def _():
        val_engine.run(val_loader, max_epochs=1)

    @train_engine.on(Events.EPOCH_STARTED)
    def _():
        # test_engine.run()
        exit(0)

    train_engine.run(train_loader, max_epochs=1)

    logger.info(f"Finish training at: {datetime.today()}.\n")
