import unittest

import torch

from models import ChannelAttention, SpacialAttention, CBAM


class TestModelOutputShape(unittest.TestCase):
    def setUp(self) -> None:
        self.channel_attention = ChannelAttention(32)
        self.spacial_attention = SpacialAttention()
        self.cbam = CBAM(32)

    def test_channel_attention_shape(self):
        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = self.channel_attention(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 32, 256, 256))
        )

    def test_spacial_attention_shape(self):
        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = self.spacial_attention(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 1, 256, 256))
        )

    def test_cbam_shape(self):
        input_tensor = torch.rand(5, 32, 256, 256)
        output_tensor: torch.Tensor = self.cbam(input_tensor)
        self.assertEqual(
            output_tensor.shape,
            torch.Size((5, 32, 256, 256))
        )
