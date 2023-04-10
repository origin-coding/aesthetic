import unittest

import torch

from common import TensorData
from models import ChannelAttention, SpacialAttention, CBAM, MTAesthetic


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

    def test_model(self):
        model = MTAesthetic(channels=512, kernel_size=3)
        input_tensor = TensorData(
            binary=torch.rand(5, 3, 256, 256),
            score=torch.rand(5, 3, 256, 256),
            attribute=torch.rand(5, 3, 256, 256)
        )
        output_tensor: TensorData = model(input_tensor)
        print(output_tensor["binary"])
        print(output_tensor["score"].shape)
        print(output_tensor["attribute"].shape)
