import sys
from unittest import TestCase

import torch
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from common import base_path, TensorData
from ui import process_output


class TestUI(TestCase):
    """对于桌面应用程序进行简单的测试"""
    def test_qml_initialization(self):
        """用于测试QML文件能否正常加载"""
        app = QGuiApplication(sys.argv)
        qml_file = base_path / "ui" / "main.qml"
        engine = QQmlApplicationEngine()
        engine.load(qml_file)
        self.assertTrue(engine.rootObjects())
        app.exit(0)

    def test_process_output(self):
        """用于测试模型的输出能否被正常处理为便于展示的格式"""
        output_tensor = TensorData(
            binary=torch.tensor(0.3), score=torch.rand(10), attribute=torch.rand(11)
        )

        processed_data = process_output(output_tensor)
        self.assertIsInstance(processed_data["binary"], bool)
        self.assertIsInstance(processed_data["score"], float)
        self.assertIsInstance(processed_data["attribute"]["symmetry"], bool)  # 这里只测试一个即可
