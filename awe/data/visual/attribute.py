"""Implementation of all visual attributes."""

import dataclasses
from typing import Any, Callable, Generic, Optional, Tuple, TypeVar, Union

import awe.data.graph.dom
import awe.data.visual.context
import awe.data.visual.structs
import awe.utils

T = TypeVar('T')
TInput = TypeVar('TInput')

def parse_font_family(value: str):
    """Parses CSS font family string into a list of font families."""

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
    """Parses border CSS properties into a tuple for all border sides."""

    # The visual extractor gives only one value if it's the same for all sides.
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
    """Transforms RGB color into a normalized feature vector."""

    return [v.hue, v.brightness / 255, v.alpha / 255]

def select_image(v: str):
    """
    Transforms image into a categorical feature, ignoring the actual URL, but
    not e.g., linear gradient.
    """

    return 'url' if v.startswith('url') else v

def select_decoration(v: str):
    """
    Transforms `text-decoration` to include only the first value (decoration
    style), but not the others (line, thickness, etc.).
    """

    return v.split(maxsplit=1)[0]

def select_border(v: list[str]):
    """Transforms border value from string with units to actual number."""

    def get_pixels(token: str):
        if token == 'none':
            return 0
        if token.endswith('px'):
            return float(token[:-2])
        raise RuntimeError(f'Cannot parse pixels from "{token}".')

    return [get_pixels(s.split(maxsplit=1)[0]) for s in v]

def select_shadow(v: str):
    """
    Transforms `text-shadow` into a categorical feature ignoring RGB values, but
    not offset.
    """

    return 'rgb' if v.startswith('rgb') else v

def select_z_index(v: str):
    """
    Transforms `z-index` into a categorical feature ignoring the actual value,
    but not `auto`.
    """

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
    """
    Represents one visual attribute of JSON DOM data type `TInput` and parsed
    Python type `T`.
    """

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
        """Default parsed value for given `node`."""

        if callable(self.default):
            return self.default(node)
        return self.default

    def parse(self, value: TInput, node_data: dict[str, Any]):
        """Parses JSON `value` into Python type `T`."""

        if self.complex_parser is None:
            return self._simple_parse(value)

        values = {
            k[len(self.name):]: self._check_value(v)
            for k, v in node_data.items()
            if k.startswith(self.name)
        }
        return self.complex_parser(values, self.default)

    def _check_value(self, value: TInput):
        """Checks JSON `value` against `load_types`."""

        if not isinstance(value, self.load_types):
            raise RuntimeError(f'Expected attribute "{self.name}" to be ' +
                f'loaded as {self.load_types} but found {type(value)} ' +
                f'({repr(value)}).')
        return value

    def _simple_parse(self, value: TInput):
        return self.parser(self._check_value(value))

    def prepare(self, c: AttributeContext):
        """Prepares this visual feature on the training set."""

    # pylint: disable-next=unused-argument
    def get_out_dim(self, extraction: awe.data.visual.context.Extraction):
        """
        Computes output dimension of this attribute's feature vector (returned
        by `compute`).
        """

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
        """Computes feature vector."""

        return self.select(c)

    def _select(self, v: T):
        """Transforms parsed value `v` into a final value."""

        if self.selector is None:
            return [v]
        return self.selector(v)

    def get_value_or_default(self, c: AttributeContext):
        """Gets default parsed value if one has not been loaded from visuals."""

        return (
            c.get_value(self) or
            self.parse(self.get_default(c.node), node_data={})
        )

    def select(self, c: AttributeContext):
        """Obtains transformed parsed value or a default one."""

        return self._select(self.get_value_or_default(c))

class CategoricalAttribute(VisualAttribute):
    """Categorical visual attribute."""

    def select(self, c: AttributeContext):
        """Transforms parsed value into the final category name."""

        v = self.get_value_or_default(c)
        if self.selector is None:
            return v
        return self.selector(v)

    def prepare(self, c: AttributeContext):
        # Determine how many unique categories there are.
        i = c.extraction.categorical[self.name][self.select(c)]
        i.count += 1

    def get_out_dim(self, extraction: awe.data.visual.context.Extraction):
        return len(extraction.categorical[self.name])

    def compute(self, c: AttributeContext):
        # Return one-hot encoded category.
        d = c.extraction.categorical[self.name]
        i = d.get(self.select(c))
        r = [0] * len(d)
        if i is not None:
            r[i.unique_id - 1] = 1
        return r

class MinMaxAttribute(VisualAttribute):
    """Numerical visual attribute that is min-max scaled."""

    def prepare(self, c: AttributeContext):
        # Determine min and max values in the training data.
        values = self.select(c)
        c.extraction.update_values(self.name, values)

    def compute(self, c: AttributeContext):
        # Perform min-max scaling.
        values = self.select(c)
        min_values = c.extraction.min_values[self.name]
        max_values = c.extraction.max_values[self.name]
        return [
            (val - min_val) / (max_val - min_val)
            for val, min_val, max_val in zip(values, min_values, max_values)
        ]

_VISUAL_ATTRIBUTES: list[VisualAttribute[Any, Any]] = [
    CategoricalAttribute('font_family', parser=parse_font_family,
        default='"Times New Roman"'),
    MinMaxAttribute('font_size', load_types=(float, int), default=16),
        # In pixels.
    MinMaxAttribute('font_weight', lambda v: [v / 100],
        parser=float, default='400'),
        # In font weight units divided by 100. E.g., "normal" is 4.
    CategoricalAttribute('font_style', default='normal'),
    CategoricalAttribute('text_decoration', select_decoration, default='none'),
    CategoricalAttribute('text_align', parser=parse_prefixed, default='start'),
    VisualAttribute('color', **COLOR, default='#000000ff'),
    VisualAttribute('background_color', **COLOR, default='#00000000'),
    CategoricalAttribute('background_image', select_image, default='none'),
    MinMaxAttribute('border', **BORDER, default='none'),
    CategoricalAttribute('box_shadow', select_shadow, default='none'),
    CategoricalAttribute('cursor', default='auto'),
    MinMaxAttribute('letter_spacing', load_types=(float, int), default=0),
        # In pixels.
    MinMaxAttribute('line_height', load_types=(float, int),
        default=lambda n: n.visuals['font_size'] * 1.2),
        # In pixels.
    VisualAttribute('opacity', load_types=(str, int), parser=float, default=1),
        # 0 = transparent, 1 = opaque.
    MinMaxAttribute('outline', **BORDER, default='none'),
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
