import jax
from julax.base import dispatch
from .base import LayerBase

from rich.console import Console
from rich.tree import Tree
from rich.markup import escape

import humanize
import zlib

from julax.layers import Repeat

console = Console()


def string_color(s: str) -> str:
    return f"color({(zlib.adler32(s.encode()) + 123) % 256})"


def colorof(layer: LayerBase) -> str:
    return string_color(layer.__class__.__name__)


def nameof(layer: LayerBase) -> str:
    s = layer.__class__.__name__
    return f"[{colorof(layer)}]{s}[/]"


def argsof(layer: LayerBase) -> str:
    fields = layer.model_dump(
        exclude_defaults=True, exclude_unset=True, exclude_none=True
    )
    for k, v in type(layer).model_fields.items():
        if k in fields and v.repr is False:
            del fields[k]
        if isinstance(getattr(layer, k), LayerBase):
            del fields[k]

    if fields:
        args = [f"[bright_blue]{k}[/]={escape(repr(v))}" for k, v in fields.items()]
        return "(" + ", ".join(args) + ")"
    else:
        return ""


def param_info(layer: LayerBase) -> str:
    num_params, num_states = layer.numel()
    total = num_params + num_states
    if total == 0:
        return ""
    s = f" [dim]# Total Params: [bright_green]{humanize.intcomma(total)}[/][/]"
    if num_states == 0:
        return s
    else:
        return (
            s
            + f" [trainable=[bright_green]{humanize.intcomma(num_params)}[/], non_trainable=[bright_green]{humanize.intcomma(num_states)}[/]]"
        )


def summary(layer: LayerBase) -> str:
    return f"[bold]{nameof(layer)}[/][dim]{argsof(layer)}[/]"


@dispatch
def to_rich(layer: Repeat) -> Tree:
    root = Tree(summary(layer) + param_info(layer), guide_style=colorof(layer))
    child = to_rich(layer.layer)
    child.label = f"[{colorof(layer.layer)}]0..{layer.n - 1}[/] [bright_yellow]=>[/] {child.label}"
    root.children.append(child)
    return root


@dispatch
def to_rich(layer: LayerBase) -> Tree:
    root = Tree(summary(layer) + param_info(layer), guide_style=colorof(layer))
    for name, sublayer in layer.sublayers().items():
        child = to_rich(sublayer)
        child.label = (
            f"[{colorof(sublayer)}]{name}[/] [bright_yellow]=>[/] {summary(sublayer)}"
        )
        child.guide_style = colorof(sublayer)
        root.children.append(child)
    return root


@dispatch
def to_rich(x: jax.Array) -> str:
    return str(jax.typeof(x))


@dispatch
def to_rich(t: dict) -> Tree:
    root = Tree("")
    for k, v in t.items():
        child = to_rich(v)
        if isinstance(child, Tree):
            if child.label:
                child.label = f"[bright_blue]{k}[/] [bright_yellow]=>[/] {child.label}"
            else:
                child.label = f"[bright_blue]{k}[/]"
            root.children.append(child)
        elif isinstance(child, str):
            root.add(f"[bright_blue]{k}[/] [bright_yellow]=>[/] {child}")
        else:
            raise NotImplementedError()
    return root


def pprint(x):
    console.print(to_rich(x))
