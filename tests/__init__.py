import unittest

from BeautifulReport import BeautifulReport

from common import base_path


def main():
    suite = unittest.defaultTestLoader.discover(start_dir=".", pattern="test_*.py")
    result = BeautifulReport(suite)
    result.report(filename="test_report.html", description="测试报告", report_dir=base_path / "outputs")


if __name__ == '__main__':
    main()
