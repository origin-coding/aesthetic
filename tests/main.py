import unittest

from BeautifulReport import BeautifulReport

from common import output_path


def test_main(filename: str, description: str) -> None:
    """
    测试过程的入口函数
    :param filename: 测试报告的文件名，存放在 aesthetic/outputs/ 目录下
    :param description: 测试报告的描述信息
    :return: None
    """
    suite = unittest.defaultTestLoader.discover(start_dir=".", pattern="test_*.py")
    result = BeautifulReport(suite)
    result.report(filename=filename, description=description, report_dir=output_path)
