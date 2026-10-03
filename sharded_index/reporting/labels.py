"""Wording of the figures in the configured language.

Every axis label, legend entry and panel title of the figures is looked up
here, so switching ``figures.language`` redraws the same figures in the other
language.  Names of strategies and algorithms stay as they are in the tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Language = Literal["ru", "en"]

_PHRASES: dict[str, dict[str, str]] = {
    "overlap_at_one": {
        "ru": "Overlap@{k} при опросе одного шарда",
        "en": "Overlap@{k} with one shard probed",
    },
    "overlap_at_budget": {
        "ru": "Overlap@{k} с полным индексом",
        "en": "Overlap@{k} with the unsharded index",
    },
    "single_shard_share": {
        "ru": "Доля запросов, покрытых одним шардом",
        "en": "Share of queries covered by one shard",
    },
    "fanout_mean": {
        "ru": "Среднее число шардов в покрытии запроса",
        "en": "Mean number of shards in the query cover",
    },
    "fanout": {"ru": "Число шардов в покрытии запроса", "en": "Shards in the query cover"},
    "share_of_queries": {"ru": "Доля запросов", "en": "Share of queries"},
    "shards_probed": {"ru": "Опрошено шардов", "en": "Shards probed"},
    "full_cover": {"ru": "полное покрытие", "en": "full cover"},
    "duplication": {
        "ru": "Дублирование документов (шардов на документ)",
        "en": "Document duplication (shards per document)",
    },
    "duplication_short": {
        "ru": "Дублирование (шардов на документ)",
        "en": "Duplication (shards per document)",
    },
    "fanout_short": {"ru": "Шардов в покрытии запроса", "en": "Shards in the query cover"},
    "volume_short": {"ru": "Объём опроса одного шарда", "en": "Probed volume, one shard"},
    "volume_at_one": {
        "ru": "Объём опроса одного шарда (доля корпуса)",
        "en": "Probed volume with one shard (share of corpus)",
    },
    "volume": {"ru": "Объём опроса (доля корпуса)", "en": "Probed volume (share of corpus)"},
    "one_shard_line": {"ru": "один шард на запрос", "en": "one shard per query"},
    "delta_overlap": {
        "ru": "Δ overlap@{k} к hash (один шард)",
        "en": "Δ overlap@{k} vs hash (one shard)",
    },
    "delta_single_shard": {
        "ru": "Δ доли запросов в один шард к hash",
        "en": "Δ single-shard share vs hash",
    },
    "delta_fanout": {
        "ru": "Δ среднего числа шардов к hash",
        "en": "Δ mean cover size vs hash",
    },
    "queries": {"ru": "запросов", "en": "queries"},
    "method": {"ru": "алгоритм", "en": "method"},
    "strategy": {"ru": "стратегия", "en": "strategy"},
    "sample": {"ru": "выборка", "en": "sample"},
    "seed": {"ru": "Зерно кластеризации", "en": "Clustering seed"},
    "train_size": {"ru": "Запросов журнала в графе", "en": "Training queries in the graph"},
    "no_parameter": {"ru": "нет параметра", "en": "no parameter"},
    "connectivity": {
        "ru": "Связность слов запроса (квартиль NPMI)",
        "en": "Connectivity of the query terms (NPMI quartile)",
    },
    "terms": {"ru": "Слов в запросе", "en": "Terms in the query"},
    "out_of_graph": {"ru": "вне графа", "en": "out of graph"},
    "single_term": {"ru": "одно слово", "en": "single term"},
    "delta_pp": {"ru": "Δ overlap@{k}, п. п.", "en": "Δ overlap@{k}, p.p."},
    "cluster_size": {"ru": "Размер кластера (слов)", "en": "Cluster size (terms)"},
    "clusters": {"ru": "Кластеров", "en": "Clusters"},
    "cluster": {"ru": "Кластер", "en": "Cluster"},
    "largest_clusters": {"ru": "{n} крупнейших кластеров", "en": "{n} largest clusters"},
    "cluster_terms": {"ru": "кластер {c} ({n} слов)", "en": "cluster {c} ({n} terms)"},
    "total_edge_weight": {"ru": "Суммарный вес рёбер", "en": "Total edge weight"},
    "terms_count": {"ru": "{n} слов", "en": "{n} terms"},
    "clusters_terms": {"ru": "{c} кластеров, {n} слов", "en": "{c} clusters, {n} terms"},
    "mean": {"ru": "в среднем", "en": "mean"},
}

_SAMPLES: dict[str, dict[str, str]] = {
    "train": {"ru": "обучающие", "en": "train"},
    "holdout": {"ru": "отложенные", "en": "holdout"},
    "ood": {"ru": "OOD", "en": "OOD"},
    "connected_pairs": {"ru": "связные пары", "en": "connected pairs"},
    "connected_triples": {"ru": "связные тройки", "en": "connected triples"},
    "frequent_pairs": {"ru": "частые пары", "en": "frequent pairs"},
    "frequent_triples": {"ru": "частые тройки", "en": "frequent triples"},
}

_PANEL_LETTERS = {"ru": "абвгдежзик", "en": "abcdefghij"}


@dataclass(frozen=True)
class Labels:
    """Phrases of the figures in one language."""

    language: Language

    def __call__(self, key: str, **values: object) -> str:
        return _PHRASES[key][self.language].format(**values)

    def sample(self, name: str) -> str:
        """Display name of a query sample; unknown names are shown as they are."""
        return _SAMPLES.get(name, {}).get(self.language, name)

    def slice(self, slicing: str, name: str) -> str:
        """Display name of a slice of the queries."""
        if slicing == "terms":
            return name
        return _PHRASES[name][self.language] if name in _PHRASES else name

    def number(self, value: int) -> str:
        """Integer with a thousands separator: a thin space in Russian, a comma in English."""
        formatted = f"{value:,}"
        return formatted.replace(",", "\u2009") if self.language == "ru" else formatted

    def panel(self, index: int) -> str:
        """Panel letter of a multi-panel figure: ``а)`` in Russian, ``a)`` in English."""
        letters = _PANEL_LETTERS[self.language]
        return f"{letters[index % len(letters)]})"
