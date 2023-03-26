import click

from tests import test_main


@click.group()
@click.pass_context
def main(_: click.Context):
    pass


@main.command()
@click.option("-f", "--filename", default="report.html", show_default=True,
              help="测试报告的文件名")
@click.option("-d", "--description", default="自动测试报告", show_default=True,
              help="测试报告的描述信息")
def test(filename: str, description: str):
    """进行自动化测试，并将测试报告输出到outputs文件夹中"""
    test_main(filename, description)


if __name__ == '__main__':
    main()
