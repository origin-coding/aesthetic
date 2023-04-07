from datetime import datetime

from .utils import setup_config, setup_logger


def train_main(config_filename: str):
    config = setup_config(filename=config_filename)
    logger = setup_logger(config)

    logger.info(f"Finish training at: {datetime.today()}.\n")
