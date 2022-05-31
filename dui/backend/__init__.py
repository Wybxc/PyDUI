from typing import Any, Generic, Protocol, TypeVar


UIControl = TypeVar("UIControl")


class UIBackend(Protocol, Generic[UIControl]):
    """UI后端。"""

    def __init__(self) -> None:
        ...

    def window(self) -> UIControl:
        """返回窗口的引用。"""
        ...

    def create(self, name: str) -> UIControl:
        """创建控件。"""
        ...

    def put(self, container: UIControl, obj: UIControl, index: int = 0) -> None:
        """将控件添加为其他控件的子控件。"""
        ...

    def release(self, container: UIControl, obj: UIControl) -> None:
        """删除控件。"""
        ...

    def modify(self, obj: UIControl, name: str, value: Any) -> None:
        """修改控件属性。"""
        ...

    def run(self) -> None:
        """运行后端。"""
        ...


backends = ["qt6"]


def get_default_backend() -> UIBackend[Any]:
    import importlib

    for backend in backends:
        try:
            return importlib.import_module(f".{backend}", package=__name__).backend()
        except ImportError:
            pass

    raise RuntimeError("No UI backend available.")
