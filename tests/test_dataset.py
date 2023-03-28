import unittest

import torch

from datasets import cuhk_pq_dataset, AADBDataset


class TestDataset(unittest.TestCase):
    def test_bin_dataset(self):
        # 测试数据集能否被正常加载
        self.assertEqual(cuhk_pq_dataset.classes, ["HighQuality", "LowQuality"])

        # 测试数据集长度
        self.assertEqual(len(cuhk_pq_dataset), 10000)
        self.assertEqual(len(cuhk_pq_dataset[0]), 2)

        # 测试数据集的标签能否被正常处理
        self.assertEqual(cuhk_pq_dataset.imgs[0][1], 0)
        self.assertEqual(cuhk_pq_dataset[0][1], 1)

        # 测试数据的类型
        self.assertIsInstance(cuhk_pq_dataset[0][0], torch.Tensor)
        self.assertIsInstance(cuhk_pq_dataset[0][1], torch.Tensor)

    def test_aadb_dataset(self):
        # 检验数据集能否正常加载
        dataset = AADBDataset()

        # 测试数据集的长度和类型
        self.assertEqual(len(dataset), 10000)
        self.assertEqual(len(dataset[0]), 2)

        # 测试数据的类型
        self.assertEqual(dataset[0][1].shape, torch.Size([11]))
