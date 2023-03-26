import unittest

import torch

from common import Task
from models import ChannelAttention, SpacialAttention, CBAM, SharedLayer, MTAesthetic


class TestModelOutputShape(unittest.TestCase):
    def test_channel_attention_shape(self):
        channel_attention = ChannelAttention(32)

        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = channel_attention(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 32, 256, 256))
        )

    def test_spacial_attention_shape(self):
        spacial_attention = SpacialAttention()

        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = spacial_attention(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 1, 256, 256))
        )

    def test_cbam_shape(self):
        cbam = CBAM(32)

        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = cbam(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 32, 256, 256))
        )

    def test_shared_layer_shape(self):
        shared_layer = SharedLayer()
        input_tensor = torch.rand(100, 3, 256, 256)
        output_tensor = shared_layer(input_tensor)

        self.assertEqual(
            output_tensor.shape,
            torch.Size((100, 1024, 16, 16))
        )

    def test_model_shape(self):
        model = MTAesthetic()
        input_tensor = torch.rand(100, 3, 256, 256)

        output_tensor_bin = model(input_tensor, Task.BINARY)
        self.assertEqual(
            output_tensor_bin.shape,
            torch.Size((100, 1))
        )

        output_tensor_score = model(input_tensor, Task.SCORE)
        self.assertEqual(
            output_tensor_score.shape,
            torch.Size((100, 1))
        )

        output_tensor_attr = model(input_tensor, Task.ATTRIBUTE)
        self.assertEqual(
            output_tensor_attr.shape,
            torch.Size((100, 11))
        )

    def test_benchmark(self):
        model = MTAesthetic()
        input_tensor = torch.rand(100, 3, 256, 256)

        # CPU上测试，60次计算用时244s，预计8分钟一个epoch，GPU上只会更少
        for i in range(1):  # 6000 // 100
            _ = model(input_tensor, Task.BINARY)
            _ = model(input_tensor, Task.SCORE)
            _ = model(input_tensor, Task.ATTRIBUTE)
