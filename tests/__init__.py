import unittest


def main():
    suite = unittest.defaultTestLoader.discover(start_dir=".", pattern="test_*.py")
    runner = unittest.TextTestRunner()
    runner.run(suite)


if __name__ == '__main__':
    main()
