import ast
import inspect
import sys
from contextlib import contextmanager
from functools import lru_cache
from itertools import count
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import Self

from .backend import UIBackend, get_default_backend

T = TypeVar("T")
P = TypeVar("P")
_sentinel = object()


class UIItem(Generic[T]):
    """UI 节点。"""

    parent: Optional["UIItem[T]"]
    slots: List["UIItem[T]"]

    def __init__(self):
        self.parent = getattr(UIContext.current, "top", None)
        self.slots = []

    @contextmanager
    def child_scope(self):
        yield

    def create(self):
        """创建控件。"""

    def modify(self, target: Self) -> None:
        """修改控件属性，以符合新的 target。"""
        pass

    def release(self):
        """释放资源。"""
        for slot in self.slots:
            slot.release()
        self.slots = []

    def match(self, other: "UIItem[T]") -> bool:
        """检查两个UIItem是否相同。"""
        return self is other


class UIItemWithId(UIItem[T]):
    """带 ID 的 UI 节点。"""

    id: int

    def __init__(self, id: int):
        super().__init__()
        self.id = id

    def match(self, other: "UIItem[T]") -> bool:
        return isinstance(other, UIItemWithId) and self.id == other.id


class UIGroup(UIItemWithId[T]):
    """UI 节点组。"""

    def __init__(self, id: int):
        super().__init__(id)


class UIState(UIItemWithId[T]):
    """UI 状态节点。"""

    value: T

    def __init__(self, id: int, default: T):
        super().__init__(id)
        self.value = default

    @overload
    def __call__(self) -> T:
        pass

    @overload
    def __call__(self, value: T) -> None:
        pass

    def __call__(self, value: Union[T, object] = _sentinel) -> Optional[T]:
        if value is _sentinel:
            return self.value
        else:
            p = self.parent
            while p and not isinstance(p, UIFunction):
                p = p.parent
            if p:
                p.render(UIContext.current)
            self.value = value


@lru_cache(maxsize=None)
def get_annotations(cls: type):
    d: Set[str] = set()
    for c in cls.mro():
        try:
            d.update(c.__annotations__.keys())
        except AttributeError:
            pass
    return d


class UIItemWithArgs(UIItemWithId[T]):
    args: Dict[str, Any]

    def __init__(self, id: int, args: Dict[str, Any]):
        super().__init__(id)
        self.args = args

    def __getattr__(self, name: str) -> Any:
        if name not in get_annotations(self.__class__) and not name.startswith("_"):
            return self.args[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in get_annotations(self.__class__) and not name.startswith("_"):
            self.args[name] = value
        else:
            object.__setattr__(self, name, value)

    def match(self, other: "UIItem[T]") -> bool:
        return (
            isinstance(other, UIItemWithArgs)
            and self.id == other.id
            # and self.args == other.args
        )


class UIComponent(UIItemWithArgs[T]):
    """UI组件。"""

    name: str
    """名称。"""
    item: T

    def __init__(
        self,
        id: int,
        name: str,
        args: Dict[str, Any],
    ):
        super().__init__(id, args)
        backend = UIContext.current.backend
        self.name = name
        self.item = backend.create(self.name)

    @contextmanager
    def child_scope(self):
        context = UIContext.current
        old = context.container, context.component_index
        context.container = self.item
        context.component_index = 0
        yield
        context.container, context.component_index = old
        context.component_index += 1

    def create(self):
        """创建控件。"""
        context = UIContext.current
        backend = context.backend
        for name, value in self.args.items():
            backend.modify(self.item, name, value)
        backend.put(context.container, self.item, context.component_index)

    def modify(self, target: Self) -> None:
        """修改控件属性，以符合新的 target。"""
        if not isinstance(target, self.__class__):
            raise TypeError(
                f"{self.__class__.__name__} not match {target.__class__.__name__}"
            )
        for name, value in target.args.items():
            if value != self.args.get(name):
                UIContext.current.backend.modify(self.item, name, value)
                self.args[name] = value

    def release(self):
        """释放资源。"""
        UIContext.current.backend.release(self.container, self.item)
        super().release()


class UIGlobalSettings(UIItemWithArgs[T]):
    """UI窗口设定组件。"""

    args: Dict[str, Any]

    def __init__(
        self,
        id: int,
        args: Dict[str, Any],
    ):
        super().__init__(id, args)

    def create(self):
        """创建控件。"""
        backend = UIContext.current.backend
        window = backend.window()
        for name, value in self.args.items():
            backend.modify(window, name, value)

    def modify(self, target: Self) -> None:
        """修改控件属性，以符合新的 target。"""
        if not isinstance(target, self.__class__):
            raise TypeError(
                f"{self.__class__.__name__} not match {target.__class__.__name__}"
            )
        backend = UIContext.current.backend
        window = backend.window()
        for name, value in target.args.items():
            if value != self.args.get(name):
                backend.modify(window, name, value)
                self.args[name] = value


class UIFunction(UIItemWithArgs[T]):
    """UI函数。"""

    func: "UIFunc"

    def __init__(self, id: int, func: "UIFunc", args: Dict[str, Any]):
        super().__init__(id, args)
        self.func = func

    def match(self, other: "UIItem[T]") -> bool:
        return (
            isinstance(other, UIFunction)
            and self.id == other.id
            and self.args == other.args
        )

    def render(self, ctx: "UIContext[T]") -> None:
        """重渲染。"""
        context = UIContext.current

        old = context.curr, context.next_index
        context.curr = self
        context.next_index = 0

        self.func(ctx, self.id, **self.args)

        context.curr, context.next_index = old

    def create(self):
        """创建控件。"""
        self.func(UIContext.current, self.id, **self.args)

    def modify(self, target: Self) -> None:
        """修改控件属性，以符合新的 target。"""
        self.args = target.args
        self.func(UIContext.current, self.id, **self.args)


class UIContext(Generic[T]):
    """UI 上下文。"""

    backend: UIBackend[T]
    """UI 后端。"""
    curr: UIItem[T]
    """当前正在渲染的 UI 节点。"""
    next_index: int
    """下一个 UI 子节点的 slot index。"""
    container: T
    """当前正在渲染的 UI 组件的容器。"""
    component_index: int
    """当前正在渲染的 UI 在容器里的 index。"""

    current: ClassVar["UIContext[Any]"]

    def __init__(self, backend: UIBackend[T]):
        UIContext.current = self
        self.backend = backend
        self.curr = UIItem()
        self.next_index = 0
        self.container = backend.window()
        self.component_index = 0

    @contextmanager
    def enter(self):
        old = self.curr, self.next_index
        self.curr = self.curr.slots[self.next_index]
        self.next_index = 0
        yield
        self.curr, self.next_index = old
        self.next_index += 1

    @contextmanager
    def use(self, item: UIItem[T]):
        if self.next_index > len(self.curr.slots):
            raise RuntimeError("No slot found")

        create = False  # 是否是新建的 Item

        if self.next_index == len(self.curr.slots):  # 如果不存在旧的 Item，则新建一个
            self.curr.slots.append(item)
            create = True
        else:
            curr = self.curr.slots[self.next_index]
            if not isinstance(curr, item.__class__):  # 如果存在旧的 Item，但是不是相同的类型，则新建一个
                self.curr.slots.insert(self.next_index, item)
                create = True
            elif not curr.match(item):  # 如果存在旧的 Item，但是不相同，则覆盖之
                self.curr.slots[self.next_index].release()
                self.curr.slots[self.next_index] = item
                create = True

        with self.enter():
            with item.child_scope():
                yield item

            if create:
                self.curr.create()
            else:
                self.curr.modify(item)

            # 检查 slots 是否结束，若未结束，删除剩余的所有元素
            while self.next_index < len(self.curr.slots):
                self.curr.slots.pop().release()

    @contextmanager
    def group(self, id: int):
        with self.use(UIGroup(id)):
            yield

    def use_state(self, id: int, default: T) -> "UIState[T]":
        with self.use(UIState(id, default)):
            return cast(UIState[T], self.curr)


class UserUI(Generic[T, P]):
    arg_names: List[str]

    def __init__(self, arg_names: List[str]):
        self.arg_names = arg_names

    def create_item(self, id: int, args: Dict[str, Any]) -> UIItem[T]:
        raise NotImplementedError()

    @contextmanager
    def __call__(self, *args: Any, **kwargs: Any):
        ctx, id, *cargs = args
        ctx = cast(UIContext[T], ctx)
        for name, value in zip(self.arg_names, cargs):
            kwargs[name] = value
        with ctx.use(self.create_item(id, kwargs)) as fc:
            yield cast(P, fc)


class UI(UserUI[T, Any]):
    """用户定义的函数式组件。"""

    def __init__(self, ui_func: Callable[..., None]):
        self.ui_func = UIFunc(ui_func)
        super().__init__(self.ui_func.arg_names)

    def compiled(self) -> str:
        return str(self.ui_func)

    def run(self):
        """作为根组件运行。"""
        backend = cast(UIBackend[T], get_default_backend())
        ctx = UIContext(backend)
        UIFunction(_unique_id(), self.ui_func, {}).render(ctx)
        backend.run()

    def create_item(self, id: int, args: Dict[str, Any]) -> UIItem[T]:
        return UIFunction(id, self.ui_func, args)


class UserUIComponent(UserUI[T, P]):
    def __init__(self, cls: Type[P]):
        arg_names = getattr(cls, "__annotations__", {}).keys()
        super().__init__(list(arg_names))
        self.name = cls.__name__

    def create_item(self, id: int, args: Dict[str, Any]) -> UIItem[T]:
        return UIComponent(id, self.name, args)


class UserUIGlobalSettings(UserUI[T, P]):
    def __init__(self, cls: Type[P]):
        arg_names = getattr(cls, "__annotations__", {}).keys()
        super().__init__(list(arg_names))
        self.name = cls.__name__

    def create_item(self, id: int, args: Dict[str, Any]) -> UIItem[T]:
        return UIGlobalSettings(id, args)


class State(Generic[T]):
    def __init__(self, default_value: T):
        self._value = default_value

    @overload
    def __call__(self) -> T:
        pass

    @overload
    def __call__(self, value: T) -> None:
        pass

    def __call__(self, value: Union[T, object] = _sentinel) -> Optional[T]:
        raise NotImplementedError


class Event:
    """事件（占位）。"""

    def __init__(self, event_name: Any):
        ...

    def __enter__(self) -> Any:
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        ...


# region AST变换

_id_gen = count()


def _unique_id() -> int:
    return next(_id_gen)


def _expr(code: str) -> ast.expr:
    return ast.parse(code, mode="eval").body


def _stmt(code: str) -> ast.stmt:
    return ast.parse(code, mode="exec").body[0]


def _name(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


def _constant(value: Any) -> ast.expr:
    return ast.Constant(value=value)


def _group_wrap(nodes: Union[List[ast.stmt], ast.stmt]) -> List[ast.stmt]:
    return [
        ast.With(
            items=[ast.withitem(context_expr=_expr(f"__ctx__.group({_unique_id()})"))],
            body=nodes if isinstance(nodes, list) else [nodes],
        )
    ]


class UIFunc:
    """函数式UI。"""

    compiled: Callable[..., Any]
    arg_names: List[str]

    def __init__(self, func: Callable[..., None]):
        _module = sys.modules[func.__module__]
        _globals = _module.__dict__.copy()
        _ast = cast(ast.FunctionDef, ast.parse(inspect.getsource(func)).body[0])

        if _ast.args.posonlyargs:
            raise TypeError("Positional only arguments are not allowed.")
        self.arg_names = [arg.arg for arg in _ast.args.args]

        _ast.args.args = [
            ast.arg(arg="__ctx__"),
            ast.arg(arg="__id__"),
        ] + _ast.args.args
        _ast.body = _group_wrap(_ast.body)
        _ast.decorator_list = [
            deco
            for deco in _ast.decorator_list
            if not (isinstance(deco, ast.Name) and deco.id == "UI")
        ]

        UIFuncTransformer(_globals).visit(_ast)

        self._ast = ast.fix_missing_locations(ast.Module(body=[_ast], type_ignores=[]))
        compiled = compile(self._ast, _module.__file__ or "<string>", "exec")
        exec(compiled, _globals)
        self.compiled = _globals[func.__name__]

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return self.compiled(*args, **kwargs)

    def __str__(self) -> str:
        return ast.unparse(self._ast)


class UIFuncTransformer(ast.NodeTransformer):
    def __init__(self, globals_: Optional[Dict[str, Any]] = None):
        self._globals = globals_ or {}

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            func = self._globals.get(node.func.id)
            if func and isinstance(func, UserUI):
                return ast.Call(
                    func=node.func,
                    args=[
                        _name("__ctx__"),
                        _constant(_unique_id()),
                        *node.args,
                    ],
                    keywords=node.keywords,
                )
            elif func and isinstance(func, type) and issubclass(func, State):
                return ast.Call(
                    func=_expr("__ctx__.use_state"),
                    args=[
                        _constant(_unique_id()),
                        *node.args,
                    ],
                    keywords=node.keywords,
                )
        return node

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        return ast.If(
            test=node.test,
            body=_group_wrap(node.body),
            orelse=_group_wrap(node.orelse),
        )

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        return _group_wrap(
            ast.For(
                target=node.target,
                iter=node.iter,
                body=_group_wrap(node.body),
                orelse=_group_wrap(node.orelse),
            )
        )

    def visit_While(self, node: ast.While):
        self.generic_visit(node)
        return _group_wrap(
            ast.While(
                test=node.test,
                body=_group_wrap(node.body),
                orelse=_group_wrap(node.orelse),
            )
        )

    def visit_With(self, node: ast.With):
        self.generic_visit(node)
        if len(node.items) == 1:
            cexpr = node.items[0].context_expr
            ovars = node.items[0].optional_vars
            params = ast.unparse(ovars) if ovars else ""
            if isinstance(cexpr, ast.Call) and isinstance(cexpr.func, ast.Name):
                func = self._globals.get(cexpr.func.id)
                if func and isinstance(func, type) and issubclass(func, Event):
                    fdef = cast(
                        ast.FunctionDef,
                        _stmt(f"def __event_handler_{_unique_id()}__({params}): ..."),
                    )
                    fdef.body = node.body
                    evt = cast(ast.Attribute, cexpr.args[0])
                    evt.ctx = ast.Store()
                    return [
                        fdef,
                        ast.Assign(
                            targets=[evt],
                            value=ast.Name(id=fdef.name, ctx=ast.Load()),
                        ),
                    ]
        return node


# endregion
