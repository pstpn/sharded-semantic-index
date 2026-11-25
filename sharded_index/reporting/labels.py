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
    "hash": {"ru": "хеш", "en": "hash"},
    "no_replication": {"ru": "без репликации", "en": "no replication"},
    "overlap_at": {
        "ru": "Качество выдачи ($Q_b$) при опросе {shards}",
        "en": "Retrieval quality ($Q_b$) with {shards} probed",
    },
    "overlap_at_budget": {
        "ru": "Качество выдачи ($Q_b$) относительно эталона",
        "en": "Retrieval quality ($Q_b$) against the reference index",
    },
    "single_shard_share": {
        "ru": "Доля запросов, покрытых одним шардом ($P_1$)",
        "en": "Share of queries covered by one shard ($P_1$)",
    },
    "fanout_mean": {
        "ru": "Среднее число шардов в покрытии запроса ($F$)",
        "en": "Mean number of shards in the query cover ($F$)",
    },
    "fanout": {
        "ru": "Число шардов в покрытии запроса ($F$)",
        "en": "Shards in the query cover ($F$)",
    },
    "share_of_queries": {"ru": "Доля запросов", "en": "Share of queries"},
    "shards_probed": {"ru": "Опрошено шардов", "en": "Shards probed"},
    "full_cover": {"ru": "полное покрытие", "en": "full cover"},
    "duplication": {
        "ru": "Дублирование документов ($D$; шардов на документ)",
        "en": "Document duplication ($D$; shards per document)",
    },
    "duplication_short": {
        "ru": "Дублирование ($D$; шардов на документ)",
        "en": "Duplication ($D$; shards per document)",
    },
    "fanout_short": {
        "ru": "Шардов в покрытии запроса ($F$)",
        "en": "Shards in the query cover ($F$)",
    },
    "volume_short": {
        "ru": "Объём опроса ($V_b$; {shards})",
        "en": "Probed volume ($V_b$; {shards})",
    },
    "volume_at": {
        "ru": "Объём опроса ($V_b$; {shards}), доля корпуса",
        "en": "Probed volume ($V_b$; {shards}), share of corpus",
    },
    "volume": {
        "ru": "Объём опроса ($V_b$; доля корпуса)",
        "en": "Probed volume ($V_b$; share of corpus)",
    },
    "one_shard_line": {"ru": "один шард на запрос", "en": "one shard per query"},
    "largest_shard_short": {
        "ru": "Крупнейший шард ($L$; доля корпуса)",
        "en": "Largest shard ($L$; share of corpus)",
    },
    "largest_shard_pct": {
        "ru": "Крупнейший шард ($L$), % корпуса",
        "en": "Largest shard ($L$), % of corpus",
    },
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
    "terms": {"ru": "Различных термов в запросе", "en": "Distinct terms in the query"},
    "out_of_graph": {"ru": "вне графа", "en": "out of graph"},
    "single_term": {"ru": "одно слово", "en": "single term"},
    "delta_pp": {
        "ru": "Изменение качества выдачи ($\\Delta Q_b$) при опросе {shards}, п. п.",
        "en": "Change in retrieval quality ($\\Delta Q_b$) with {shards} probed, p.p.",
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
