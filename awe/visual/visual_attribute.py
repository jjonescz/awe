from dataclasses import dataclass, field
from typing import (TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar,
                    Union)

from awe import utils
from awe.visual import color

if TYPE_CHECKING:
    from awe import awe_graph, features

T = TypeVar('T')

@dataclass
class AttributeContext(Generic[T]):
    """Everything needed to compute feature from `VisualAttribute` of a node."""

    attribute: 'VisualAttribute[T]'
    node: 'awe_graph.HtmlNode'
    context: 'features.RootContext'
    freezed: bool

    @property
    def value(self) -> T:
        return self.node.visuals.get(self.attribute.name)

def categorical(c: AttributeContext[str]):
    if c.freezed:
        i = c.context.visual_categorical[c.attribute.name].get(c.value)
        if i is None:
            return 0
    else:
        i = c.context.visual_categorical[c.attribute.name][c.value]
        i.count += 1
    return [i.unique_id]

def select_color(c: AttributeContext[color.Color]):
    return [c.value.hue, c.value.brightness / 255, c.value.alpha / 255]

COLOR = {
    'selector': select_color,
    'parser': color.Color.parse,
    'labels': ['hue', 'brightness', 'alpha']
}

@dataclass
class VisualAttribute(Generic[T]):
    name: str
    """Name in snake_case."""

    selector: Callable[[AttributeContext[T]], list[float]] = \
        field(repr=False)
    """Converts attribute to feature vector."""

    parser: Callable[[Any], T] = field(default=lambda x: x, repr=False)
    """Used when converting from JSON value to Python value."""

    labels: Optional[list[str]] = field(default=None)
    """Column labels of the resulting feature vector."""

    default: Union[T, Callable[['awe_graph.HtmlNode'], T]] = \
        field(default=None, repr=False)
    """Default JSON value if DOM data are missing this attribute."""

    @property
    def camel_case_name(self):
        utils.to_camel_case(self.name)

    def get_default(self, node: 'awe_graph.HtmlNode') -> T:
        if callable(self.default):
            return self.default(node)
        return self.default

    def get_labels(self):
        return self.labels or [self.name]

_VISUAL_ATTRIBUTES: list[VisualAttribute[Any]] = [
    VisualAttribute('font_family', categorical, default='"Times New Roman"'),
    VisualAttribute('font_size', lambda c: [c.value or 0], default=16),
        # In pixels.
    VisualAttribute('font_weight', lambda c: [float(c.value) / 100],
        default='400'),
        # In font weight units divided by 100. E.g., "normal" is 4.
    VisualAttribute('font_style', categorical, default='normal'),
    VisualAttribute('text_decoration', categorical, default='none'),
    VisualAttribute('text_align', categorical, default='start'),
    VisualAttribute('color', **COLOR, default='#000000ff'),
    VisualAttribute('background_color', **COLOR, default='#00000000'),
    VisualAttribute('background_image', categorical, default='none'),
    VisualAttribute('box_shadow', categorical, default='none'),
    VisualAttribute('cursor', categorical, default='auto'),
    VisualAttribute('letter_spacing', lambda c: [c.value], default=0),
        # In pixels.
    VisualAttribute('line_height', lambda c: [c.value],
        default=lambda n: n.visuals['font_size'] * 1.2),
        # In pixels.
    VisualAttribute('opacity', lambda c: [c.value], default=1),
        # 0 = transparent, 1 = opaque.
    VisualAttribute('overflow', categorical, default='auto'),
    VisualAttribute('pointer_events', categorical, default='auto'),
    VisualAttribute('text_shadow', categorical, default='none'),
    VisualAttribute('text_overflow', categorical, default='clip'),
    VisualAttribute('text_transform', categorical, default='none'),
    VisualAttribute('z_index', categorical, default='auto'),
]

VISUAL_ATTRIBUTES = {
    a.name: a
    for a in _VISUAL_ATTRIBUTES
}
