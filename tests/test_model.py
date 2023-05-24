import unittest

import torch

from common import TensorData
from models import ChannelAttention, SpacialAttention, CBAM, MTAesthetic, SharedLayer
from models import MTLoss, MTDwa


class TestModelOutput(unittest.TestCase):
    """对模型的输出格式及类型进行测试"""

    def test_channel_attention_output(self):
        """检验Channel Attention模块能否正确地输出对应格式的数据"""
        channel_attention = ChannelAttention(32)

        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = channel_attention(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 32, 256, 256))
        )

    def test_spacial_attention_output(self):
        """检验Special Attention模块能否正确地输出对应格式的数据"""
        spacial_attention = SpacialAttention()

        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = spacial_attention(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 1, 256, 256))
        )

    def test_cbam_output(self):
        """检验CBAM模块能否正确地输出对应格式的数据"""
        cbam = CBAM(32)

        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = cbam(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 32, 256, 256))
        )

    def test_shared_layer_output(self):
        """检验参数共享层的输出类型和格式是否正确"""
        input_tensor = torch.rand(5, 3, 256, 256)
        model = SharedLayer(1024, 3)
        output_tensor = model(input_tensor)

        self.assertIsInstance(output_tensor, torch.Tensor)
        self.assertEqual(output_tensor.shape, torch.Size((5, 1024, 8, 8)))

    def test_model_output(self):
        """检验模型整体的输入输出格式是否正确"""
        model = MTAesthetic(channels=512, kernel_size=3)
        input_tensor = TensorData(
            binary=torch.rand(5, 3, 256, 256), score=torch.rand(5, 3, 256, 256), attribute=torch.rand(5, 3, 256, 256)
        )
        output_tensor: TensorData = model(input_tensor)

        self.assertEqual(output_tensor["binary"].shape, torch.Size((5,)))
        self.assertEqual(output_tensor["score"].shape, torch.Size((5, 10)))
        self.assertEqual(output_tensor["attribute"].shape, torch.Size((5, 11)))

    def test_loss_output(self):
        """用于测试损失函数 MTLoss 能否正确地返回结果"""
        output_tensor = label_tensor = TensorData(
            binary=torch.rand(5, ), score=torch.rand(5, 10), attribute=torch.rand(5, 11)
        )

        loss_fn = MTLoss()
        loss = loss_fn(output_tensor, label_tensor)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_dwa_output(self) -> None:
        """用于测试损失函数 MTDwa 能否正确地返回结果"""
        output_tensor = label_tensor = TensorData(
            binary=torch.rand(5, ), score=torch.rand(5, 10), attribute=torch.rand(5, 11)
        )

        loss_fn = MTDwa()
        loss = loss_fn(output_tensor, label_tensor)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
