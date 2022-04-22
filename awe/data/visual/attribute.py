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
class AttributeContext:
    """Everything needed to compute feature from `VisualAttribute` of a node."""

    node: awe.data.graph.dom.Node
    extraction: awe.data.visual.context.Extraction

    def get_value(self, attr: 'VisualAttribute[T]') -> T:
        return self.node.visuals.get(attr.name)

    def set_value(self, attr: 'VisualAttribute[T]', v: T):
        self.node.visuals[attr.name] = v

def select_color(v: awe.data.visual.structs.Color):
    return [v.hue, v.brightness / 255, v.alpha / 255]

def select_image(v: str):
    return 'url' if v.startswith('url') else v

def select_decoration(v: str):
    return v.split(maxsplit=1)[0]

def select_border(v: list[str]):
    def get_pixels(token: str):
        if token == 'none':
            return 0
        if token.endswith('px'):
            return float(token[:-2])
        raise RuntimeError(f'Cannot parse pixels from "{token}".')

    return [get_pixels(s.split(maxsplit=1)[0]) for s in v]

def select_shadow(v: str):
    return 'rgb' if v.startswith('rgb') else v

def select_z_index(v: str):
    return 'number' if v.isdecimal() else v

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

    selector: Optional[Callable[[T], list[float]]] = \
        dataclasses.field(default=None, repr=False)
    """Converts Python value to feature vector."""

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

    default: Union[TInput, Callable[[awe.data.graph.dom.Node], TInput]] = \
        dataclasses.field(default=None, repr=False)
    """Default JSON value if DOM data are missing this attribute."""

    @property
    def camel_case_name(self):
        awe.utils.to_camel_case(self.name)

    def get_default(self, node: awe.data.graph.dom.Node) -> TInput:
        if callable(self.default):
            return self.default(node)
        return self.default

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

    def prepare(self, c: AttributeContext):
        """Prepares feature during training."""

    # pylint: disable-next=unused-argument
    def get_out_dim(self, extraction: awe.data.visual.context.Extraction):
        if self.complex_parser is not None:
            raise ValueError(
                'Unable to determine dimension of an attribute with ' +
                f'a complex parser ({self.name!r}).')
        if callable(self.default):
            if self.selector is not None:
                raise ValueError(
                    'Unable to determine dimension of an attribute with ' +
                    'a custom selector and dynamic default value ' +
                    f'({self.name!r}).')
            return 1
        return len(self._select(self._simple_parse(self.default)))

    def compute(self, c: AttributeContext) -> list[float]:
        """Computes feature after all training data have been prepared."""
        return self._select(c.get_value(self))

    def _select(self, v: T):
        if self.selector is None:
            return [v]
        return self.selector(v)

class CategoricalAttribute(VisualAttribute):
    def select(self, c: AttributeContext):
        v = c.get_value(self)
        if self.selector is None:
            return v
        return self.selector(v)

    def prepare(self, c: AttributeContext):
        i = c.extraction.categorical[self.name][self.select(c)]
        i.count += 1

    def get_out_dim(self, extraction: awe.data.visual.context.Extraction):
        return len(extraction.categorical[self.name])

    def compute(self, c: AttributeContext):
        d = c.extraction.categorical[self.name]
        i = d.get(self.select(c))
        r = [0] * len(d)
        if i is not None:
            r[i.unique_id - 1] = 1
        return r

_VISUAL_ATTRIBUTES: list[VisualAttribute[Any, Any]] = [
    CategoricalAttribute('font_family', parser=parse_font_family,
        default='"Times New Roman"'),
    VisualAttribute('font_size', lambda v: [v or 0],
        load_types=(float, int), default=16),
        # In pixels.
    VisualAttribute('font_weight', lambda v: [v / 100],
        parser=float, default='400'),
        # In font weight units divided by 100. E.g., "normal" is 4.
    CategoricalAttribute('font_style', default='normal'),
    CategoricalAttribute('text_decoration', select_decoration, default='none'),
    CategoricalAttribute('text_align', parser=parse_prefixed, default='start'),
    VisualAttribute('color', **COLOR, default='#000000ff'),
    VisualAttribute('background_color', **COLOR, default='#00000000'),
    CategoricalAttribute('background_image', select_image, default='none'),
    VisualAttribute('border', **BORDER, default='none'),
    CategoricalAttribute('box_shadow', select_shadow, default='none'),
    CategoricalAttribute('cursor', default='auto'),
    VisualAttribute('letter_spacing', load_types=(float, int), default=0),
        # In pixels.
    VisualAttribute('line_height', load_types=(float, int),
        default=lambda n: n.visuals['font_size'] * 1.2),
        # In pixels.
    VisualAttribute('opacity', load_types=(str, int), parser=float, default=1),
        # 0 = transparent, 1 = opaque.
    VisualAttribute('outline', **BORDER, default='none'),
    CategoricalAttribute('overflow', default='auto'),
    CategoricalAttribute('pointer_events', default='auto'),
    CategoricalAttribute('text_shadow', select_shadow, default='none'),
    CategoricalAttribute('text_overflow', default='clip'),
    CategoricalAttribute('text_transform', default='none'),
    CategoricalAttribute('z_index', select_z_index, default='auto'),
]

VISUAL_ATTRIBUTES = {
    a.name: a
    for a in _VISUAL_ATTRIBUTES
}
