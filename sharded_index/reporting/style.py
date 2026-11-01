"""Visual language of the figures.

The look follows pgfplots, the plotting engine of the LaTeX reports, set in
Times New Roman as the journal requires: a framed plot area with a light major
grid, inward ticks, filled markers on saturated lines and a boxed legend.
Colour carries one identity per figure: the clustering method in the overview
figures, the strategy family in the per-method figures.  Replicas darken the
family hue; hash baselines are grey, hollow and dashed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from sharded_index.config import Strategy

PAGE_WIDTH = 6.5
"""Width of a full-width figure in inches: the text width of an A4 article."""
COLUMN_WIDTH = 3.3
"""Width of a one-column figure in inches."""

BLUE, RED, GREEN, PURPLE, YELLOW, CYAN, GREY = (
    "#0000B3",
    "#B30000",
    "#007A00",
    "#7A0099",
    "#B38600",
    "#007A99",
    "#808080",
)
"""Saturated TikZ-like hues: ``blue!70!black``, ``red!70!black`` and their kin."""
HASH_COLOR = "#808080"
INK_MUTED = "#808080"
SURFACE = "#ffffff"
GRID_COLOR = "#D9D9D9"
DIVERGING = "RdBu_r"
SEQUENTIAL = "Blues"

METHOD_COLORS = {"leiden": BLUE, "infomap": RED, "cpm": GREEN, "metis": PURPLE}
METHOD_MARKERS = {"leiden": "o", "infomap": "s", "cpm": "^", "metis": "D"}
SPARE_COLORS = (CYAN, YELLOW, GREY)
SPARE_MARKERS = ("v", "P", "X", "*")
FAMILY_COLORS = {"base": BLUE, "aff": RED, "bal": GREEN}
FAMILY_MARKERS = {"base": "o", "aff": "s", "bal": "^"}
REPLICA_SHADE = {1: 0.0, 2: 0.25, 3: 0.5}
"""How much darker than the family hue a strategy with that many replicas is drawn."""

FONT_FAMILY = "Times New Roman"
"""Body face of the journal; STIX (a Times design with Cyrillic) stands in where it is missing."""


def _theme() -> dict[Any, Any]:
    return {
        "font.family": "serif",
        "font.serif": [FONT_FAMILY, "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "mathtext.fallback": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.linewidth": 0.6,
        "axes.edgecolor": "black",
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.axisbelow": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.4,
        "grid.linestyle": "-",
        "axes.prop_cycle": cycler(color=[BLUE, RED, GREEN, PURPLE, CYAN, YELLOW, GREY]),
        "axes.unicode_minus": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.0,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "black",
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.2,
        "legend.handletextpad": 0.5,
        "patch.linewidth": 0.6,
        "figure.titlesize": 9,
        "figure.titleweight": "normal",
        "figure.dpi": 150,
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    }


THEME = _theme()

F = TypeVar("F", bound=Callable[..., Any])


def themed(function: F) -> F:
    """Run a plotting function with the figure theme applied."""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logging.getLogger("fontTools").setLevel(logging.WARNING)
        with mpl.rc_context(THEME):
            return function(*args, **kwargs)

    return cast(F, wrapper)


def save(figure: Figure, path: Path) -> None:
    """Write a PDF that does not change between runs (no creation date)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", metadata={"CreationDate": None})


def categorical_x(axis: Any) -> None:
    """Drop the minor ticks of an axis whose positions are categories, not numbers."""
    axis.tick_params(axis="x", which="minor", bottom=False, top=False)


def _assign(names: Iterable[str], known: dict[str, str], spare: tuple[str, ...]) -> dict[str, str]:
    names = list(names)
    assigned = {name: known[name] for name in names if name in known}
    free = [value for value in spare if value not in assigned.values()]
    for name in names:
        if name not in assigned:
            assigned[name] = free.pop(0) if free else spare[-1]
    return assigned


def method_colors(methods: Iterable[str]) -> dict[str, str]:
    """Colour of every clustering method: known names keep theirs, others take the spare ones."""
    return _assign(methods, METHOD_COLORS, SPARE_COLORS)


def method_markers(methods: Iterable[str]) -> dict[str, str]:
    """Marker of every clustering method, assigned like the colours."""
    return _assign(methods, METHOD_MARKERS, SPARE_MARKERS)


def shade(color: str, amount: float) -> tuple[float, float, float]:
    """The colour blended towards black (``amount`` > 0) or white (``amount`` < 0)."""
    target = (0.0, 0.0, 0.0) if amount > 0 else (1.0, 1.0, 1.0)
    weight = abs(amount)
    red, green, blue = to_rgb(color)
    return tuple(
        channel * (1 - weight) + end * weight
        for channel, end in zip((red, green, blue), target, strict=True)
    )  # type: ignore[return-value]


def strategy_color(name: str) -> Any:
    """Family hue, darkened by the replica count; grey for a hash baseline."""
    strategy = Strategy.parse(name)
    if strategy.is_hash:
        return HASH_COLOR
    return shade(FAMILY_COLORS[strategy.family], REPLICA_SHADE.get(strategy.replicas, 0.6))


def strategy_marker(name: str) -> str:
    return FAMILY_MARKERS[Strategy.parse(name).family]


def strategy_line_style(name: str) -> Literal["-", "--"]:
    return "--" if Strategy.parse(name).is_hash else "-"


def marker_style(name: str, color: Any, marker: str | None = None) -> dict[str, Any]:
    """Marker keyword arguments: filled for a semantic strategy, hollow for its hash baseline."""
    hollow = Strategy.parse(name).is_hash
    return {
        "marker": marker or strategy_marker(name),
        "markerfacecolor": SURFACE if hollow else color,
        "markeredgecolor": color,
    }
