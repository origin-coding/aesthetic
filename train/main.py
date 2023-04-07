from .utils import setup_config, setup_logger


def train_main(config_filename: str):
    config = setup_config(filename=config_filename)
    _logger = setup_logger(config)
