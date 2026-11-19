"""Complete comparisons: strategy colours and method markers on shared axes."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from sharded_index.config import Config, Strategy
from sharded_index.paths import Paths
from sharded_index.pipeline.common import setup
from sharded_index.reporting.labels import Labels
from sharded_index.reporting.style import (
    comparison_legend,
    comparison_marker,
    method_markers,
    save,
    themed,
)


def matrix(
    frame: pd.DataFrame,
    methods: list[str],
    strategies: list[str],
    category: str,
    categories: list[str],
    metric: str,
) -> Any:
    """Align every configured measurement; never silently omit a configuration."""
    keys = ["method", "strategy", category]
    index = pd.MultiIndex.from_product([methods, strategies, categories], names=keys)
    values = frame.set_index(keys)[metric].reindex(index).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        message = f"Missing or non-finite measurements for {metric}"
        raise ValueError(message)
    return values.reshape(len(methods) * len(strategies), len(categories))


@themed
def draw(
    panels: list[tuple[str, list[str], Any, bool]],
    methods: list[str],
    strategies: list[str],
    path: Path,
    text: Labels,
) -> None:
    """Show all configurations, with horizontal offsets only to separate markers."""

    def metric_label(title: str, proportion: bool) -> str:
        first, separator, rest = title.partition("\n")
        return first + (", %" if proportion else "") + separator + rest

    columns = 2 if len(panels) > 2 else 1
    rows = (len(panels) + columns - 1) // columns
    figure = Figure(figsize=(9 if columns == 2 else 7, 2.7 * rows + 1), layout="constrained")
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    markers = method_markers(methods)
    parsed = [Strategy.parse(strategy) for strategy in strategies]
    offsets = np.linspace(-0.40, 0.40, len(strategies) * len(methods))
    for axis, (title, labels, values, proportion) in zip(axes, panels, strict=False):
        for method_index, method in enumerate(methods):
            for strategy_index, strategy in enumerate(parsed):
                data = values[method_index * len(strategies) + strategy_index]
                offset = offsets[strategy_index * len(methods) + method_index]
                axis.scatter(
                    np.arange(len(labels)) + offset,
                    data * 100 if proportion else data,
                    zorder=3,
                    **comparison_marker(strategy, markers[method]),
                )
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.set_xlim(-0.55, len(labels) - 0.45)
        axis.set_ylabel(metric_label(title, proportion), fontsize=8)
        axis.grid(axis="x", visible=False)
        if proportion:
            lower = np.floor(values.min() * 20) * 5 if "\n" in title else 0
            axis.set_ylim(lower - 3, 103)
        else:
            axis.set_ylim(bottom=0)
    for axis in axes[len(panels) :]:
        axis.set_visible(False)
    figure.legend(handles=comparison_legend(markers, text), loc="outside lower center", ncol=5)
    save(figure, path)


def run(config: Config, paths: Paths) -> None:
    methods = list(config.clustering.methods)
    strategies = list(config.partition.strategies)
    samples = list(config.queries.sample_names())
    routing = pd.read_csv(paths.metrics / "routing.csv")
    for language in config.figures.languages:
        text = Labels(language)
        sample_panels = [
            (
                text(metric),
                [text.sample(s) for s in samples],
                matrix(routing, methods, strategies, "sample", samples, metric),
                proportion,
            )
            for metric, proportion in [("single_shard_share", True), ("fanout_mean", False)]
        ]
        draw(
            sample_panels,
            methods,
            strategies,
            paths.figures / language / "overview_by_sample.pdf",
            text,
        )


if __name__ == "__main__":
    run(*setup())
