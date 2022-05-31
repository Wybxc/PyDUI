import sys
from typing import cast
from PySide6 import QtWidgets, QtCore

app = QtWidgets.QApplication(sys.argv)

w = QtWidgets.QWidget()
w.resize(250, 150)
w.setWindowTitle("PyDUI")

l = QtWidgets.QVBoxLayout()
w.setLayout(l)
layout = QtWidgets.QVBoxLayout()


label = QtWidgets.QLabel("Hello PyDUI!")
label.setAlignment(cast(QtCore.Qt.Alignment, QtCore.Qt.AlignCenter))
layout.insertWidget(0, label)
button = QtWidgets.QPushButton("Click Me")
layout.insertWidget(1, button)

l.insertLayout(0, layout)

count = 0


def on_click():
    global count
    count += 1
    if count == 1:
        label.setText("Clicked 1 time!")
    else:
        label.setText(f"Clicked {count} times!")


button.clicked.connect(on_click)  # type: ignore

w.show()
app.exec()
