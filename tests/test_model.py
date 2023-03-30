import unittest

import torch

from models import ChannelAttention, SpacialAttention, CBAM, SharedLayer


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
