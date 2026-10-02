"""The documentation must cover every setting of the configuration."""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from sharded_index.config import Config
from sharded_index.paths import PROJECT_ROOT

CONFIGURATION = (PROJECT_ROOT / "docs" / "configuration.md").read_text()
FREE_FORM = {"ablations.variants", "sensitivity.parameters"}
"""Settings whose values are user-defined mappings, documented as a whole."""


def _models(annotation: Any) -> Iterator[tuple[type[BaseModel], bool]]:
    """Models inside an annotation and whether they sit in a name-keyed dictionary."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        yield annotation, False
    elif get_origin(annotation) is dict:
        for model, _ in _models(get_args(annotation)[1]):
            yield model, True
    elif get_origin(annotation) in (Union, types.UnionType):
        for argument in get_args(annotation):
            yield from _models(argument)


def setting_paths(model: type[BaseModel], prefix: str = "") -> Iterator[str]:
    """Dotted paths of every leaf setting, e.g. ``clustering.methods.<name>.algorithm``."""
    for name, field in model.model_fields.items():
        path = f"{prefix}{name}"
        nested = [] if path in FREE_FORM else list(_models(field.annotation))
        if not nested:
            yield path
        for section, keyed in nested:
            yield from setting_paths(section, f"{path}.<name>." if keyed else f"{path}.")


def test_every_setting_is_documented() -> None:
    paths = list(setting_paths(Config))
    assert "clustering.methods.<name>.resolution" in paths
    assert "queries.ood.column" in paths
    missing = [path for path in paths if f"`{path}`" not in CONFIGURATION]
    assert not missing, f"not described in docs/configuration.md: {missing}"


def test_documented_values_match_the_project_config(project_config: Config) -> None:
    assert f"{len(project_config.text.stop_words)} слово" in CONFIGURATION
    assert f"{len(project_config.graph.stop_words)} слов" in CONFIGURATION
    assert f"{len(project_config.partition.strategies)} стратегий" in CONFIGURATION
    assert f"{len(project_config.ablations.variants)} вариантов" in CONFIGURATION
    assert f"{len(project_config.evaluation.comparisons.pairs)} пар" in CONFIGURATION
    for name in project_config.ablations.variants:
        assert f"`{name}`" in CONFIGURATION, name
