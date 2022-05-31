from typing import Any
from dui import UI, State, Event
from dui.components import Button, Label, VBox, Window # type: ignore


@UI
def Counter(count: int, **kwargs: Any):
    with Label(**kwargs) as label:
        label.text = "Clicked 1 time!" if count == 1 else f"Clicked {count} times!"


@UI
def App():
    with Window() as w:
        w.title = "PyDUI"
        w.size = (250, 150)

    with VBox():
        count = State(0)

        if count() == 0:
            with Label("Hello PyDUI!", alignment="center"):
                pass
        else:
            with Counter(count(), alignment="center"):
                pass

        with Button(text="Click Me") as btn:
            with Event(btn.on_click):
                count(count() + 1)

with open('App.py', 'w') as f:
    f.write(App.compiled())

App.run()
