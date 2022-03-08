import dataclasses
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar, Union

import awe.data.graph.dom
import awe.data.visual.context
import awe.data.visual.structs
import awe.utils

T = TypeVar('T')
TInput = TypeVar('TInput')

def parse_font_family(value: str):
    values = [
        v.strip().strip('"').strip()
        for v in value.split(',')
    ]
    values = [v for v in values if len(v) != 0]
    return values[0].lower() if len(values) != 0 else ''

def parse_prefixed(value: str):
    """Trims vendor prefixes from CSS value (e.g., `-webkit-left` -> `left`)."""
    PREFIX = '-webkit-' # only Chrome should be enough for computed CSS
    if value.startswith(PREFIX):
        return value[len(PREFIX):]
    return value

BORDER_SIDES = ['left', 'top', 'right', 'bottom']

def parse_border(values: dict[str, str], default: str):
    value = values.get('')
    if value is not None:
        return [value] * 4
    return [values.get(side, default) for side in BORDER_SIDES]

@dataclasses.dataclass
class AttributeContext(Generic[T]):
    """Everything needed to compute feature from `VisualAttribute` of a node."""

    attribute: 'VisualAttribute[T]'
    node: awe.data.graph.dom.Node
    extraction: awe.data.visual.context.Extraction
    freezed: bool

    @property
    def value(self) -> T:
        return self.node.visuals.get(self.attribute.name)

    @value.setter
    def value(self, v: T):
        self.node.visuals[self.attribute.name] = v

def categorical(c: AttributeContext[str]):
    if c.freezed:
        i = c.extraction.categorical[c.attribute.name].get(c.value)
        if i is None:
            return [0]
    else:
        i = c.extraction.categorical[c.attribute.name][c.value]
        i.count += 1
    return [i.unique_id]

def select_color(c: AttributeContext[awe.data.visual.structs.Color]):
    return [c.value.hue, c.value.brightness / 255, c.value.alpha / 255]

def select_image(c: AttributeContext[str]):
    c.value = 'url' if c.value.startswith('url') else c.value
    return categorical(c)

def select_decoration(c: AttributeContext[str]):
    c.value = c.value.split(maxsplit=1)[0]
    return categorical(c)

def select_border(c: AttributeContext[list[str]]):
    def get_pixels(token: str):
        if token == 'none':
            return 0
        if token.endswith('px'):
            return float(token[:-2])
        raise RuntimeError(f'Cannot parse pixels from "{token}".')

    return [get_pixels(v.split(maxsplit=1)[0]) for v in c.value]

def select_shadow(c: AttributeContext[str]):
    if c.value.startswith('rgb'):
        c.value = 'rgb'
    return categorical(c)

def select_z_index(c: AttributeContext[str]):
    if c.value.isdecimal():
        c.value = 'number'
    return categorical(c)

COLOR = {
    'selector': select_color,
    'parser': awe.data.visual.structs.Color.parse,
    'labels': ['hue', 'brightness', 'alpha']
}

BORDER = {
    'selector': select_border,
    'complex_parser': parse_border,
    'labels': BORDER_SIDES
}

@dataclasses.dataclass
class VisualAttribute(Generic[T, TInput]):
    name: str
    """Name in snake_case."""

    selector: Optional[Callable[[AttributeContext[T]], list[float]]] = \
        dataclasses.field(default=None, repr=False)
    """Converts attribute to feature vector."""

    parser: Callable[[TInput], T] = dataclasses.field(default=lambda x: x, repr=False)
    """Used when converting from JSON value to Python value."""

    complex_parser: Optional[Callable[[dict[str, TInput], TInput], T]] = \
        dataclasses.field(default=None, repr=False)
    """
    Like parser but gets all node's DOM data prefixed with attribute's name.
    """

    load_types: Union[type[TInput], Tuple[type[TInput]]] = \
        dataclasses.field(default=str, repr=False)
    """What types are allowed to be loaded from JSON DOM data."""

    labels: Optional[list[str]] = dataclasses.field(default=None)
    """Column labels of the resulting feature vector."""

    default: Union[T, Callable[[awe.data.graph.dom.Node], T]] = \
        dataclasses.field(default=None, repr=False)
    """Default JSON value if DOM data are missing this attribute."""

    @property
    def camel_case_name(self):
        awe.utils.to_camel_case(self.name)

    def get_default(self, node: awe.data.graph.dom.Node) -> T:
        if callable(self.default):
            return self.default(node)
        return self.default

    def get_labels(self):
        if self.labels is not None:
            return [f'{self.name}_{l}' for l in self.labels]
        return [self.name]

    def parse(self, value: TInput, node_data: dict[str, Any]):
        if self.complex_parser is None:
            return self._simple_parse(value)

        values = {
            k[len(self.name):]: self._check_value(v)
            for k, v in node_data.items()
            if k.startswith(self.name)
        }
        return self.complex_parser(values, self.default)

    def _check_value(self, value: TInput):
        if not isinstance(value, self.load_types):
            raise RuntimeError(f'Expected attribute "{self.name}" to be ' + \
                f'loaded as {self.load_types} but found {type(value)} ' + \
                f'({repr(value)}).')
        return value

    def _simple_parse(self, value: TInput):
        return self.parser(self._check_value(value))

    def select(self, c: AttributeContext[T]):
        if self.selector is None:
            return [c.value]
        return self.selector(c)

_VISUAL_ATTRIBUTES: list[VisualAttribute[Any, Any]] = [
    VisualAttribute('font_family', categorical, parse_font_family,
        default='"Times New Roman"'),
    VisualAttribute('font_size', lambda c: [c.value or 0],
        load_types=(float, int), default=16),
        # In pixels.
    VisualAttribute('font_weight', lambda c: [c.value / 100],
        parser=float, default='400'),
        # In font weight units divided by 100. E.g., "normal" is 4.
    VisualAttribute('font_style', categorical, default='normal'),
    VisualAttribute('text_decoration', select_decoration, default='none'),
    VisualAttribute('text_align', categorical, parse_prefixed, default='start'),
    VisualAttribute('color', **COLOR, default='#000000ff'),
    VisualAttribute('background_color', **COLOR, default='#00000000'),
    VisualAttribute('background_image', select_image, default='none'),
    VisualAttribute('border', **BORDER, default='none'),
    VisualAttribute('box_shadow', select_shadow, default='none'),
    VisualAttribute('cursor', categorical, default='auto'),
    VisualAttribute('letter_spacing', load_types=(float, int), default=0),
        # In pixels.
    VisualAttribute('line_height', load_types=(float, int),
        default=lambda n: n.visuals['font_size'] * 1.2),
        # In pixels.
    VisualAttribute('opacity', load_types=(str, int), parser=float, default=1),
        # 0 = transparent, 1 = opaque.
    VisualAttribute('outline', **BORDER, default='none'),
    VisualAttribute('overflow', categorical, default='auto'),
    VisualAttribute('pointer_events', categorical, default='auto'),
    VisualAttribute('text_shadow', select_shadow, default='none'),
    VisualAttribute('text_overflow', categorical, default='clip'),
    VisualAttribute('text_transform', categorical, default='none'),
    VisualAttribute('z_index', select_z_index, default='auto'),
]

VISUAL_ATTRIBUTES = {
    a.name: a
    for a in _VISUAL_ATTRIBUTES
}