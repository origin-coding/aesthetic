import click

from tests import test_main
from train import train_main
# from ui import ui_main


@click.group()
@click.pass_context
def main(_: click.Context):
    pass


@main.command()
@click.option("-f", "--filename", default="report.html", show_default=True,
              help="测试报告的文件名，要求是HTML格式")
@click.option("-d", "--description", default="自动测试报告", show_default=True,
              help="测试报告的描述信息")
def test(filename: str, description: str):
    """进行自动化测试，并将测试报告输出到outputs文件夹中"""
    test_main(filename, description)


# @main.command()
# def ui():
#     """启动应用界面"""
#     ui_main()


@main.command()
@click.option("-f", "--filename", default="config.json", show_default=True,
              help="配置文件的文件名，要求是JSON格式")
def train(filename: str):
    """训练模型，并将结果和日志保存起来"""
    train_main(filename)


if __name__ == '__main__':
    main()
