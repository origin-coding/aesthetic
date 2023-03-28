import unittest

from BeautifulReport import BeautifulReport

from common import base_path


def test_main(filename: str, description: str):
    suite = unittest.defaultTestLoader.discover(start_dir=".", pattern="test_*.py")
    result = BeautifulReport(suite)
    result.report(filename=filename, description=description, report_dir=base_path / "outputs")
