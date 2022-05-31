from abc import abstractmethod
import sys
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, cast
from PySide6 import QtWidgets, QtCore

T = TypeVar("T")


controls: Dict[str, type] = {}


def control(name: str):
    def decorator(cls: Type[T]) -> Type[T]:
        controls[name] = cls
        return cls

    return decorator


class UIControl:
    def add(self, item: "UIControl", index: int = 0) -> None:
        pass

    def remove(self, item: "UIControl") -> None:
        pass


class UIWidget(UIControl):
    @property
    @abstractmethod
    def widget(self) -> QtWidgets.QWidget:
        pass


class UILayout(UIControl):
    @property
    @abstractmethod
    def layout(self) -> QtWidgets.QLayout:
        pass


class Window(UIWidget, UILayout):
    def __init__(self) -> None:
        self._widget = QtWidgets.QWidget()
        self._layout = QtWidgets.QVBoxLayout()
        self._widget.setLayout(self._layout)

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._widget

    @property
    def layout(self) -> QtWidgets.QLayout:
        return self._layout

    def add(self, item: UIControl, index: int = 0) -> None:
        """添加子控件。"""
        if isinstance(item, UIWidget):
            self._layout.insertWidget(index, item.widget)            
        elif isinstance(item, UILayout):
            self._layout.insertLayout(index, item.layout)

    def remove(self, item: UIControl) -> None:
        """移除子控件。"""
        if isinstance(item, UIWidget):
            self._layout.removeWidget(item.widget)
        elif isinstance(item, UILayout):
            self._layout.removeItem(item.layout)

    @property
    def title(self) -> str:
        return self._widget.windowTitle()

    @title.setter
    def title(self, value: str) -> None:
        self._widget.setWindowTitle(value)

    @property
    def size(self) -> Tuple[int, int]:
        return self._widget.size().width(), self._widget.size().height()

    @size.setter
    def size(self, value: Tuple[int, int]) -> None:
        self._widget.resize(value[0], value[1])

    @property
    def width(self) -> int:
        return self._widget.width()

    @width.setter
    def width(self, value: int) -> None:
        return self._widget.resize(value, self._widget.height())

    @property
    def height(self) -> int:
        return self._widget.height()

    @height.setter
    def height(self, value: int) -> None:
        return self._widget.resize(self._widget.width(), value)


@control("Label")
class Label(UIWidget):
    def __init__(self):
        self._label = QtWidgets.QLabel("")

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._label

    @property
    def text(self) -> str:
        return self._label.text()

    @text.setter
    def text(self, value: str) -> None:
        self._label.setText(value)

    alignments = {
        "left": QtCore.Qt.AlignLeft,
        "right": QtCore.Qt.AlignRight,
        "center": QtCore.Qt.AlignCenter,
        "justify": QtCore.Qt.AlignJustify,
    }

    @property
    def alignment(self) -> str:
        for key, value in self.alignments.items():
            if cast(QtCore.Qt.AlignmentFlag, self._label.alignment()) == value:
                return key
        return "left"

    @alignment.setter
    def alignment(self, value: str) -> None:
        self._label.setAlignment(cast(QtCore.Qt.Alignment, self.alignments[value]))


@control("Button")
class Button(UIWidget):
    def __init__(self):
        self._button = QtWidgets.QPushButton("")
        self._on_click: Optional[Callable[[], None]] = None

    @property
    def widget(self) -> QtWidgets.QWidget:
        return self._button

    @property
    def text(self) -> str:
        return self._button.text()

    @text.setter
    def text(self, value: str) -> None:
        self._button.setText(value)

    @property
    def on_click(self) -> Optional[Callable[[], None]]:
        return self._on_click

    @on_click.setter
    def on_click(self, value: Callable[[], None]) -> None:
        if self._on_click:
            self._button.clicked.disconnect(self._on_click)  # type: ignore
        self._on_click = value
        self._button.clicked.connect(value)  # type: ignore


@control("VBox")
class VBox(UILayout):
    def __init__(self):
        self._layout = QtWidgets.QVBoxLayout()

    @property
    def layout(self) -> QtWidgets.QLayout:
        return self._layout

    def add(self, item: UIControl, index: int = 0) -> None:
        """添加子控件。"""
        if isinstance(item, UIWidget):
            self._layout.insertWidget(index, item.widget)            
        elif isinstance(item, UILayout):
            self._layout.insertLayout(index, item.layout)

    def remove(self, item: UIControl) -> None:
        """移除子控件。"""
        if isinstance(item, UIWidget):
            self._layout.removeWidget(item.widget)
        elif isinstance(item, UILayout):
            self._layout.removeItem(item.layout)


class QtBackend:
    """Qt6后端。"""

    def __init__(self):
        self._app = QtWidgets.QApplication(sys.argv)
        self._window = Window()

    def window(self) -> UIControl:
        """返回窗口的引用。"""
        return self._window

    def create(self, name: str) -> UIControl:
        """创建控件。"""
        return controls[name]()

    def put(self, container: UIControl, obj: UIControl, index: int = 0) -> None:
        """将控件添加为其他控件的子控件。"""
        container.add(obj, index=index)

    def release(self, container: UIControl, obj: UIControl) -> None:
        """删除控件。"""
        container.remove(obj)

    def modify(self, obj: UIControl, name: str, value: Any) -> None:
        """修改控件属性。"""
        setattr(obj, name, value)

    def run(self) -> None:
        """运行后端。"""
        self._window.widget.show()
        self._app.exec()


backend = QtBackend
