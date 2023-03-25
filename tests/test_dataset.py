import unittest

from datasets import bin_dataset


class TestDataset(unittest.TestCase):
    def test_bin_dataset_classes(self):
        self.assertEqual(bin_dataset.classes, ["HighQuality", "LowQuality"])
