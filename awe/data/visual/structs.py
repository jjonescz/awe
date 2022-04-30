"""Structures holding visual attribute values."""

import colorsys
import dataclasses


@dataclasses.dataclass
class BoundingBox:
    """Coordinates of a node on rendered page."""

    x: int
    y: int
    width: int
    height: int

    @property
    def center_point(self):
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def top_left(self):
        return self.x, self.y

    @property
    def top_right(self):
        return self.x + self.width, self.y

    @property
    def bottom_left(self):
        return self.x, self.y + self.height

    @property
    def bottom_right(self):
        return self.x + self.width, self.y + self.height

    @property
    def corners(self):
        return self.top_left, self.top_right, self.bottom_left, self.bottom_right

    @property
    def is_positive(self):
        return self.bottom_right > (0, 0)

    def as_tuple(self):
        return self.x, self.y, self.width, self.height

@dataclasses.dataclass
class Color:
    """Color in RGBA format."""

    red: int
    green: int
    blue: int

    alpha: int
    """Alpha channel (0 = fully transparent, 255 = fully opaque)."""

    @property
    def hsv(self):
        return colorsys.rgb_to_hsv(self.red, self.green, self.blue)

    @property
    def hue(self):
        return self.hsv[0]

    @property
    def brightness(self):
        return self.hsv[2]

    @classmethod
    def parse(cls, s: str):
        """Parses color from its hex representation (e.g., `#00112233`)."""

        def h(i: int):
            return int(s[i:(i + 2)], 16)
        return Color(h(1), h(3), h(5), h(7))
