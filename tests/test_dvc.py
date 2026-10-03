"""The pipeline definition must agree with the code: a stage declares the modules it imports."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from sharded_index.paths import PROJECT_ROOT

PLUMBING = {"sharded_index/config.py", "sharded_index/paths.py", "sharded_index/pipeline/common.py"}
"""Modules that only read settings and locate files: used by every stage, computing nothing."""


def _module_file(name: str) -> Path | None:
    path = PROJECT_ROOT.joinpath(*name.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _imports(path: Path, package: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            names.add(module)
            names.update(f"{module}.{alias.name}" for alias in node.names)
    return {name for name in names if name.startswith("sharded_index")}


def imported_modules(entry: str) -> set[str]:
    """Project modules reachable from ``entry`` through imports, as paths from the project root.

    Plumbing modules are neither reported nor followed.
    """
    seen: set[Path] = set()
    todo = [entry]
    while todo:
        name = todo.pop()
        path = _module_file(name)
        if path is None or path in seen or str(path.relative_to(PROJECT_ROOT)) in PLUMBING:
            continue
        seen.add(path)
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        todo.extend(_imports(path, package))
    return {str(path.relative_to(PROJECT_ROOT)) for path in seen if path.name != "__init__.py"}


def stages() -> list[tuple[str, str, list[str]]]:
    """``(stage, entry module, declared code dependencies)`` of every stage of dvc.yaml."""
    spec = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text())
    rows = []
    for name, stage in spec["stages"].items():
        body = stage.get("do", stage)
        entry = re.search(r"-m (sharded_index\.[\w.]+)", body["cmd"])
        assert entry is not None, name
        deps = [dep if isinstance(dep, str) else next(iter(dep)) for dep in body.get("deps", [])]
        rows.append(
            (name, entry.group(1), [dep for dep in deps if dep.startswith("sharded_index")])
        )
    return rows


STAGES = stages()


@pytest.mark.parametrize(("stage", "entry", "declared"), STAGES, ids=[row[0] for row in STAGES])
def test_stage_declares_the_modules_it_imports(stage: str, entry: str, declared: list[str]) -> None:
    needed = imported_modules(entry)

    def covers(dep: str, module: str) -> bool:
        return module == dep or module.startswith(f"{dep}/")

    missing = sorted(module for module in needed if not any(covers(d, module) for d in declared))
    stale = sorted(dep for dep in declared if not any(covers(dep, m) for m in needed))
    assert not missing, f"{stage}: imported but not declared in dvc.yaml: {missing}"
    assert not stale, f"{stage}: declared in dvc.yaml but not imported: {stale}"
