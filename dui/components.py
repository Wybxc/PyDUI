from typing import Callable, Literal, Optional, Tuple
from .core import UserUIComponent, UserUIGlobalSettings


@UserUIComponent
class Label:
    text: str
    alignment: Literal["left", "right", "center", "justify"]


@UserUIComponent
class Button:
    text: str
    on_click: Optional[Callable[[], None]]


@UserUIComponent
class VBox:
    pass


@UserUIGlobalSettings
class Window:    
    title: str
    size: Tuple[int, int]
