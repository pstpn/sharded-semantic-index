# Данные и литература

## Данные

- [MS MARCO v2.1](https://huggingface.co/datasets/microsoft/ms_marco) — используется
  в экспериментах (HuggingFace `microsoft/ms_marco`, конфиг `v2.1`, train-сплит);
- [Natural Questions (open)](https://huggingface.co/datasets/google-research-datasets/nq_open) —
  запросы другого источника для выборки `ood`;
- [Yandex search](https://www.kaggle.com/datasets/sergunow/yandex-search) —
  рассматривался как кандидат, в экспериментах не задействован.

## Литература для позиционирования

- Risvik et al., *Maguro, a system for indexing and searching over very large
  text collections*, WSDM 2013 — мотивация терм-ориентированного шардирования
  (важно: наш hash-бейзлайн — это нулевая модель случайного разбиения термов,
  а не реконструкция Maguro).
- Kulkarni, Callan, *Selective Search* (topical sharding + resource selection:
  ReDDE, Taily, Rank-S) — ближайший родственник архитектуры: там кластеризуют
  документы при 1x хранения и приближённой маршрутизации; у нас — термы, точная
  маршрутизация и плата дупликацией.
- Moffat, Webber, Zobel — классика term-partitioned индексов (постинг-листы
  делятся между узлами, документы не реплицируются) — чем наша схема НЕ является.
