import unittest

import torch

from common import TrainData
from datasets import cuhk_pq_dataset, AADBDataset, AVADataset, MTAestheticDataset


class TestDataset(unittest.TestCase):
    def test_cuhk_pq_dataset(self):
        """检验CUHK-PQ数据集能否正常地被加载"""

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

        self.assertEqual(cuhk_pq_dataset[0][0].shape, torch.Size([3, 256, 256]))

    def test_aadb_dataset(self):
        """测试AADB数据集能否被正常地加载"""

        # 检验数据集能否正常加载
        dataset = AADBDataset()

        # 测试数据的类型
        self.assertIsInstance(dataset[0][0], torch.Tensor)
        self.assertIsInstance(dataset[0][1], torch.Tensor)

        self.assertEqual(cuhk_pq_dataset[0][0].shape, torch.Size([3, 256, 256]))
        self.assertEqual(dataset[0][1].shape, torch.Size([11]))

        # 测试数据集的长度
        self.assertEqual(len(dataset), 10000)
        self.assertEqual(len(dataset[0]), 2)

    def test_ava_dataset(self):
        """测试AVA数据集能否被正常地加载"""
        dataset = AVADataset()

        # 测试数据集长度
        self.assertEqual(len(dataset), 10000)
        self.assertEqual(len(dataset[0]), 2)

        # 测试数据集的类型
        self.assertIsInstance(dataset[0][0], torch.Tensor)
        self.assertIsInstance(dataset[0][1], torch.Tensor)

        self.assertEqual(cuhk_pq_dataset[0][0].shape, torch.Size([3, 256, 256]))
        self.assertEqual(dataset[0][1].shape, torch.Size([10]))

    def test_mt_aesthetic_dataset(self):
        """测试整体数据集能否被正常地加载"""
        dataset = MTAestheticDataset()
        self.assertEqual(len(dataset), 10000)
        self.assertIsInstance(dataset[0], dict)
