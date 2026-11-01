"""Wording of the figures in the configured language.

Every axis label, legend entry and panel letter of the figures is looked up
here, so switching ``figures.languages`` redraws the same figures in the other
language.  Names of strategies, algorithms and query samples stay as they are
in the tables; the figures carry no titles — what a figure shows belongs to
its caption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Language = Literal["ru", "en"]

_PHRASES: dict[str, dict[str, str]] = {
    "overlap_at": {
        "ru": "Overlap@{k} при опросе {shards}",
        "en": "Overlap@{k} with {shards} probed",
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
    "volume_short": {"ru": "Объём опроса {shards}", "en": "Probed volume, {shards}"},
    "volume_at": {
        "ru": "Объём опроса {shards} (доля корпуса)",
        "en": "Probed volume with {shards} (share of corpus)",
    },
    "volume": {"ru": "Объём опроса (доля корпуса)", "en": "Probed volume (share of corpus)"},
    "one_shard_line": {"ru": "один шард на запрос", "en": "one shard per query"},
    "largest_shard_short": {
        "ru": "Крупнейший шард (доля корпуса)",
        "en": "Largest shard (share of corpus)",
    },
    "largest_shard_pct": {"ru": "Крупнейший шард, % корпуса", "en": "Largest shard, % of corpus"},
    "budget_line": {"ru": "бюджет объёма шарда", "en": "shard-volume budget"},
    "resolution": {"ru": "Разрешение γ", "en": "Resolution γ"},
    "n_parts": {"ru": "Число частей k", "en": "Number of parts k"},
    "no_parameter": {"ru": "Без параметра", "en": "No parameter"},
    "queries": {"ru": "запросов", "en": "queries"},
    "method": {"ru": "алгоритм", "en": "method"},
    "semantic_strategy": {"ru": "семантическая стратегия", "en": "semantic strategy"},
    "hash_baseline": {"ru": "hash-разбиение", "en": "hash baseline"},
    "sample": {"ru": "выборка", "en": "sample"},
    "connectivity": {
        "ru": "Связность слов запроса (квартиль NPMI)",
        "en": "Connectivity of the query terms (NPMI quartile)",
    },
    "terms": {"ru": "Слов в запросе", "en": "Terms in the query"},
    "out_of_graph": {"ru": "вне графа", "en": "out of graph"},
    "single_term": {"ru": "одно слово", "en": "single term"},
    "delta_pp": {
        "ru": "Δ overlap@{k} при опросе {shards}, п. п.",
        "en": "Δ overlap@{k} with {shards} probed, p.p.",
    },
    "cluster_size": {"ru": "Размер кластера (слов)", "en": "Cluster size (terms)"},
    "clusters": {"ru": "Кластеров", "en": "Clusters"},
    "cluster": {"ru": "Кластер", "en": "Cluster"},
    "cluster_terms": {"ru": "кластер {c} ({n} слов)", "en": "cluster {c} ({n} terms)"},
    "total_edge_weight": {"ru": "Суммарный вес рёбер", "en": "Total edge weight"},
}

_SAMPLES = {
    "train": "train",
    "holdout": "holdout",
    "ood": "OOD",
    "connected_pairs": "connected pairs",
    "connected_triples": "connected triples",
    "frequent_pairs": "frequent pairs",
    "frequent_triples": "frequent triples",
}
"""Display names of the query samples: English in both languages, like the names in the tables."""

_PANEL_LETTERS = {"ru": "абвгдежзик", "en": "abcdefghij"}


@dataclass(frozen=True)
class Labels:
    """Phrases of the figures in one language."""

    language: Language

    def __call__(self, key: str, **values: object) -> str:
        return _PHRASES[key][self.language].format(**values)

    def shards(self, count: int) -> str:
        """``count`` shards as words: «одного шарда», «2 шардов»; "one shard", "2 shards"."""
        if self.language == "ru":
            return "одного шарда" if count == 1 else f"{count} шардов"
        return "one shard" if count == 1 else f"{count} shards"

    def sample(self, name: str) -> str:
        """Display name of a query sample; unknown names are shown as they are."""
        return _SAMPLES.get(name, name)

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
