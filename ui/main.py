import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


from .context import Context


def ui_main() -> None:
    """
    封装好的桌面应用程序的入口
    :return: None
    """

    # 创建Qt应用程序，加载QML文件
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(qml_file)

    # 将上下文注入到引擎中，使之能被QML页面使用
    context = Context()
    engine.rootContext().setContextProperty("context", context)

    # 检查是否成功加载并运行程序
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
