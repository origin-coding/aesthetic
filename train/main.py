from .config import setup_config


def train_main(config_filename: str):
    config = setup_config(filename=config_filename)
    print(config.optimizer.value)
