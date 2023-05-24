from datetime import datetime
from pathlib import Path
from unittest import TestCase

import torch
from ignite.metrics import Metric, Loss, Accuracy

from common import base_path, TrainData, TensorData, log_path
from train import setup_config, Configuration, OptimizerConfiguration
from train import setup_metrics, prepare_batch, setup_logger, setup_data


class TestTrain(TestCase):
    """对于训练过程中的一些功能进行测试"""

    def test_config(self):
        """用于训练配置能否被正常加载"""
        config_file: Path = base_path / "config.json"
        if not config_file.exists():
            return

        config = setup_config()

        self.assertIsInstance(config, Configuration)
        self.assertIsInstance(config.optimizer, OptimizerConfiguration)

    def test_metrics(self):
        """用于测试训练过程中的Metrics是否符合要求"""
        metrics = setup_metrics(device=torch.device("cpu"), config=setup_config())
        self.assertEqual(len(metrics), 5)

        key: str
        value: Metric
        for key, value in metrics.items():
            if "loss" in key:
                self.assertIsInstance(value, Loss)
            else:
                self.assertIsInstance(value, Accuracy)

    def test_prepare_batch(self):
        """用于测试prepare_batch方法能否正常工作"""
        train_data = TrainData(
            input_tensor=TensorData(
                binary=torch.rand(1, 3, 256, 256), score=torch.rand(1, 3, 256, 256),
                attribute=torch.rand(1, 3, 256, 256),
            ),
            label_tensor=TensorData(
                binary=torch.rand(1, 3, 256, 256), score=torch.rand(1, 3, 256, 256),
                attribute=torch.rand(1, 3, 256, 256),
            )
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        prepared_data = prepare_batch(train_data, device)
        self.assertIsInstance(prepared_data, dict)
        self.assertEqual(prepared_data["input_tensor"]["binary"].device, device)
        self.assertEqual(prepared_data["input_tensor"]["score"].device, device)
        self.assertEqual(prepared_data["input_tensor"]["attribute"].device, device)
        self.assertEqual(prepared_data["label_tensor"]["binary"].device, device)
        self.assertEqual(prepared_data["label_tensor"]["score"].device, device)
        self.assertEqual(prepared_data["label_tensor"]["attribute"].device, device)

    @classmethod
    def setUpClass(cls):
        for file in log_path.iterdir():
            file.unlink(missing_ok=True)

    def test_logger(self):
        """测试logger能否正常初始化。"""
        _ = setup_logger(setup_config(), test=True)
        logger_path = log_path / f"{datetime.today().date()}.log"
        self.assertTrue(logger_path.exists())

    def test_setup_data(self):
        """检验DataLoader能否正常创建和加载"""
        import warnings
        warnings.filterwarnings("ignore")

        config = setup_config()
        train_loader, val_loader, test_loader = setup_data(config)

        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)

        self.assertEqual(train_loader.batch_size, config.batch_size)
        self.assertEqual(val_loader.batch_size, config.batch_size)
        self.assertEqual(test_loader.batch_size, config.batch_size)
