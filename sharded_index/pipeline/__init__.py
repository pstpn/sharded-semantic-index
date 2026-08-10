"""DVC pipeline stages (see dvc.yaml); each runs as ``python -m sharded_index.pipeline.<stage>``.

Stages, in order:

1. :mod:`.extract_pairs`    — MS MARCO → нормализованные пары запрос-документ;
2. :mod:`.build_partitions` — NPMI-граф по train-запросам → партиции и реплики;
3. :mod:`.build_indices`    — документы по шардам + Whoosh-индексы;
4. :mod:`.evaluate`         — метрики на holdout → metrics/;
5. :mod:`.figures`          — все фигуры → reports/figures/.

Параметры — в ``params.yaml``; пути — в :mod:`sharded_index.config`
(переопределяются ``SSI_ROOT`` / ``SSI_PARAMS`` для тестов).
"""
