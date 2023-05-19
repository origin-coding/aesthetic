import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine


from .context import Context


def ui_main():
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    qml_file = Path(__file__).resolve().parent / "main.qml"
    engine.load(qml_file)

    context = Context()
    engine.rootContext().setContextProperty("context", context)

    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
