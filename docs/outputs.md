# Справочник выходных файлов

Результаты эксперимента — таблицы CSV в каталоге [`metrics/`](../metrics) и графики в
[`reports/figures/`](../reports/figures); они хранятся в git. Промежуточные артефакты лежат в
`data/` и `models/` и хранятся в кэше DVC.

Общие правила для таблиц:

- одна строка — одна конфигурация; имена столбцов одинаковы во всех файлах;
- `method` — имя алгоритма кластеризации из `clustering.methods`, `strategy` — имя стратегии из
  `partition.strategies`, `sample` — имя выборки запросов;
- доли записаны числами от 0 до 1, дробные значения округлены до 4 знаков;
- пустая ячейка означает «неприменимо»;
- запросы, по которым полный индекс ничего не находит, в метрики не входят;
- `<b>` в имени столбца — бюджет: число опрашиваемых шардов из `evaluation.budgets` или `full`
  (полное покрытие).

## Содержание

- [Таблицы результатов](#таблицы-результатов): [`partitions`](#metricspartitionscsv),
  [`routing`](#metricsroutingcsv), [`retrieval`](#metricsretrievalcsv),
  [`comparisons`](#metricscomparisonscsv), [`slices`](#metricsslicescsv),
  [`sensitivity`](#metricssensitivitycsv), [`ablations`](#metricsablationscsv),
  [`verification`](#metricsverificationcsv)
- [Описание данных](#описание-данных): [`corpus`](#metricscorpusjson),
  [`clusterings`](#metricsclusteringscsv), [`clusters`](#metricsclusterscsv),
  [`frame_words`](#metricsframe_wordscsv)
- [Графики](#графики)
- [Промежуточные артефакты](#промежуточные-артефакты)

## Таблицы результатов

Все таблицы этого раздела пишет стадия `report`.

### `metrics/partitions.csv`

Свойства разбиения, не зависящие от запросов. Строка — алгоритм и стратегия.

| Столбец | Значение |
|---|---|
| `method`, `strategy` | алгоритм и стратегия |
| `shards` | число непустых шардов |
| `replication_factor` | среднее число шардов на слово (1 — без реплик) |
| `replicated_terms` | число слов, лежащих более чем в одном шарде |
| `duplication` | среднее число шардов, в которых лежит документ; во столько раз суммарный объём шардов больше корпуса |
| `expected_duplication` | теоретическое дублирование при случайном разбиении на столько же шардов: среднее по документам `S·(1 − (1 − 1/S)^m)`, где `m` — число разных слов документа. Только для `hash_*` |
| `largest_shard` | доля документов корпуса в самом крупном шарде |
| `top10_shards` | суммарный объём десяти крупнейших шардов в долях корпуса (может быть больше 1) |
| `hash_terms` | число слов, получивших шард по хешу, а не по кластеру. Для `hash_*` пусто |
| `replica_cost_top1pct`, `replica_cost_top10pct` | доля стоимости репликации, приходящаяся на 1% и 10% самых частых слов. Стоимость реплики слова — число документов с этим словом. Только для стратегий с репликами |

### `metrics/routing.csv`

Сколько шардов нужно запросу. Строка — алгоритм, стратегия и выборка.

| Столбец | Значение |
|---|---|
| `method`, `strategy`, `sample` | алгоритм, стратегия, выборка |
| `queries` | число запросов в выборке |
| `evaluated` | число запросов с непустой выдачей полного индекса; по ним считаются все метрики |
| `terms_mean` | среднее число разных слов в запросе после удаления стоп-слов |
| `known_terms_mean` | среднее число слов запроса, которые есть в корпусе; шард есть только у них, поэтому покрытие не длиннее этого числа |
| `fanout_mean` | среднее число шардов в покрытии запроса |
| `fanout_ci_low`, `fanout_ci_high` | границы доверительного интервала среднего |
| `fanout_median`, `fanout_p95` | медиана и 95-й процентиль числа шардов в покрытии |
| `single_shard_share` | доля запросов, покрытых одним шардом |
| `single_shard_ci_low`, `single_shard_ci_high` | границы доверительного интервала этой доли |
| `hash_probe_share` | доля опросов, приходящихся на шарды hash-пространства (слова вне кластеров). Для `hash_*` пусто |
| `mixed_query_share` | доля запросов, которым нужны и шарды кластеров, и шарды hash-пространства. Для `hash_*` пусто |
| `top_shard_traffic` | доля запросов, у которых первым опрашивается один и тот же, самый востребованный шард |

### `metrics/retrieval.csv`

Качество и цена при опросе `b` шардов. Строка — алгоритм, стратегия, выборка и бюджет.

| Столбец | Значение |
|---|---|
| `method`, `strategy`, `sample` | алгоритм, стратегия, выборка |
| `budget` | число опрошенных шардов: первые `b` шардов покрытия или `full` — всё покрытие |
| `overlap` | средняя доля первых `evaluation.top_k` документов полного индекса, найденная в опрошенных шардах. При `full` равна 1 по построению |
| `overlap_ci_low`, `overlap_ci_high` | границы доверительного интервала |
| `volume` | средний объём опроса: сумма размеров опрошенных шардов в долях корпуса. Документ, лежащий в двух опрошенных шардах, учитывается дважды |
| `volume_ci_low`, `volume_ci_high` | границы доверительного интервала |
| `volume_median` | медиана объёма опроса |

### `metrics/comparisons.csv`

Парные разности двух конфигураций на одних и тех же запросах. Строка — пара, выборка и метрика.

| Столбец | Значение |
|---|---|
| `comparison` | вид сравнения: `hash` — стратегия против hash-разбиения своего семейства; `strategy` — пара стратегий из `evaluation.comparisons.pairs`; `method` — алгоритм против `reference_method` на той же стратегии |
| `method`, `strategy` | сравниваемая конфигурация |
| `baseline_method`, `baseline_strategy` | конфигурация, с которой сравнивают |
| `sample` | выборка |
| `metric` | метрика: `fanout`, `single_shard_share`, `overlap_<b>`, `volume_<b>` |
| `difference` | средняя разность «сравниваемая минус база» по запросам |
| `ci_low`, `ci_high` | границы доверительного интервала разности. Если интервал не содержит ноль, разница значима на выбранном уровне |

### `metrics/slices.csv`

Метрики на частях одной выборки (`evaluation.slices.sample`). Строка — алгоритм, стратегия и срез.

| Столбец | Значение |
|---|---|
| `method`, `strategy` | алгоритм и стратегия |
| `slicing` | способ разрезания: `connectivity` — по связности слов запроса в графе; `terms` — по числу слов запроса |
| `slice` | срез. Для `connectivity`: `Q1`…`Qn` — квантили средней связности пар слов запроса (в `Q1` наименее связные), `single_term` — запрос из одного слова графа, `out_of_graph` — в запросе есть слово вне графа. Для `terms`: `1`, `2`, … и последний срез вида `5+` |
| `queries` | число запросов в срезе |
| `connectivity_min`, `connectivity_max` | границы связности в срезе (только для `Q1`…`Qn`) |
| `terms_mean`, `known_terms_mean` | среднее число слов в запросе и число слов, имеющих шард |
| `fanout_mean`, `single_shard_share` | как в `routing.csv` |
| `overlap_<b>`, `volume_<b>` | как `overlap` и `volume` в `retrieval.csv`, по столбцу на бюджет |

### `metrics/sensitivity.csv`

Проверки устойчивости: меняется одно условие, все стратегии строятся заново и измеряются на
выборке `sensitivity.sample`. Данные готовит стадия `sensitivity`. Первая строка каждого ряда —
основная конфигурация.

| Столбец | Значение |
|---|---|
| `sweep` | что меняется: `seed` — зерно кластеризации; `resolution` или `n_parts` — собственный параметр алгоритма; `train_size` — число обучающих запросов, по которым строится граф |
| `method`, `strategy` | алгоритм и стратегия |
| `value` | значение изменяемой величины |
| `graph_terms` | число слов в графе |
| `clusters` | число кластеров |
| `queries_in_graph` | доля запросов выборки, все слова которых есть в графе |
| `shards`, `duplication`, `largest_shard` | как в `partitions.csv` |
| `fanout_mean`, `fanout_p95`, `single_shard_share` | как в `routing.csv` |
| `overlap_<b>`, `volume_<b>` | как `overlap` и `volume` в `retrieval.csv`, по столбцу на бюджет |

### `metrics/ablations.csv`

Абляции: в конфигурации меняется одна настройка, всё строится заново от графа и измеряется на
выборке `ablations.sample`. Данные готовит стадия `ablations`.

| Столбец | Значение |
|---|---|
| `variant` | имя варианта из `ablations.variants`; `baseline` — основная конфигурация |
| `method`, `strategy` | алгоритм и стратегия |
| `graph_terms`, `clusters`, `queries_in_graph` | как в `sensitivity.csv` |
| `shards`, `duplication`, `largest_shard` | как в `partitions.csv` |
| `fanout_mean`, `fanout_p95`, `single_shard_share` | как в `routing.csv` |
| `overlap_<b>`, `volume_<b>` | как `overlap` и `volume` в `retrieval.csv`, по столбцу на бюджет |

### `metrics/verification.csv`

Проверки корректности оценки. Данные готовит стадия `verify`; если хотя бы одна проверка не
прошла, стадия завершается с ошибкой и таблица не обновляется. Строка — алгоритм и стратегия.

| Столбец | Значение |
|---|---|
| `method`, `strategy` | алгоритм и стратегия |
| `assignment_exact` | раскладка документов по шардам совпала с правилом «документ лежит в каждом шарде каждого своего слова» при независимом пересчёте по текстам |
| `incomplete_covers` | число запросов, покрытие которых не содержит шарда какого-либо слова запроса (должно быть 0) |
| `queries_checked` | число проверенных запросов: все запросы всех выборок |
| `document_term_pairs` | число проверенных пар «документ — слово» |
| `ranking_order_violations` | число контрольных запросов, у которых выдача полного индекса не упорядочена по убыванию оценки (должно быть 0) |
| `ranking_prefix_mismatches` | число контрольных запросов, у которых сохранённая выдача не совпала с началом полной (должно быть 0) |
| `shards_built` | число настоящих индексов шардов, построенных для контрольных запросов |
| `documents_indexed` | число документов в этих индексах |
| `scores_compared` | число сравнений оценки документа в шарде с его оценкой в полном индексе |
| `max_score_difference` | наибольшее расхождение оценок |
| `result_mismatches` | число случаев (запрос × бюджет), когда расчётная выдача не совпала с выдачей настоящих шардов (должно быть 0) |
| `passed` | все проверки стратегии пройдены |

## Описание данных

Файлы этого раздела пишет стадия `describe`.

### `metrics/corpus.json`

| Раздел | Поля |
|---|---|
| `collection` | `documents` — документов в корпусе; `judged_relevant_documents` — из них отмечены асессором; `distractor_documents` — остальные; `queries`, `train_queries`, `holdout_queries` — уникальных запросов всего, в обучающей и в отложенной части; `judged_per_query_mean` — отмеченных пассажей на запрос |
| `vocabulary` | `terms` — слов в словаре корпуса; `document_term_pairs` — пар «документ — слово»; `terms_per_document_mean`, `terms_per_document_median` — разных слов в документе |
| `document_frequency` | `max_share` — доля документов с самым частым словом; `top10` — десять самых частых слов и их доли; `terms_over_1pct`, `terms_over_5pct` — число слов, встречающихся более чем в 1% и 5% документов |
| `graph` | `terms`, `edges` — вершин и рёбер; `components`, `largest_component_terms` — число связных компонент и размер наибольшей; `terms_in_corpus` — слов графа, встречающихся в корпусе; `corpus_vocabulary_share` — их доля в словаре корпуса; `frame_stop_words` — длина списка `graph.stop_words` |

### `metrics/clusterings.csv`

Строка — алгоритм.

| Столбец | Значение |
|---|---|
| `method` | имя алгоритма в конфиге |
| `algorithm`, `resolution`, `iterations`, `max_iterations`, `n_parts`, `seed` | настройки алгоритма |
| `graph_terms` | число кластеризованных слов |
| `clusters` | число непустых кластеров |
| `largest_cluster` | число слов в крупнейшем кластере |
| `median_cluster` | медианный размер кластера |
| `modularity` | взвешенная модулярность разбиения графа |

### `metrics/clusters.csv`

Крупнейшие кластеры каждого алгоритма (`describe.top_clusters` на алгоритм).

| Столбец | Значение |
|---|---|
| `method` | алгоритм |
| `cluster` | номер кластера |
| `terms` | число слов |
| `internal_edges` | число рёбер графа внутри кластера |
| `density` | доля пар слов кластера, соединённых ребром |
| `top_terms` | слова с наибольшей суммой весов рёбер (`describe.top_terms` слов) |

### `metrics/frame_words.csv`

Кандидаты в список `graph.stop_words`: самые частые слова обучающих запросов
(`describe.frame_word_candidates` слов), отсортированные по `ratio`. Таблица помогает
составлять список вручную: высокий `ratio` бывает и у тематических слов.

| Столбец | Значение |
|---|---|
| `term` | слово |
| `query_share` | доля обучающих запросов со словом |
| `document_share` | доля документов со словом |
| `ratio` | `query_share / document_share`: больше 1 — слово чаще встречается в запросах, чем в текстах |
| `lead_share` | доля вхождений, где слово стоит среди первых двух слов запроса |
| `listed` | слово уже есть в `graph.stop_words` |

## Графики

Стадия `figures` пишет PDF. Графики оценки построены по выборке `figures.sample`.

| Файл | Что показано |
|---|---|
| `reports/figures/methods_overlap.pdf` | overlap для каждой стратегии под каждым алгоритмом, с доверительными интервалами; ряд панелей на каждый бюджет из `figures.budgets` |
| `reports/figures/overview_single_shard_by_sample.pdf` | доля запросов, покрытых одним шардом, на каждой выборке запросов (панель на выборку) для всех стратегий и алгоритмов, с доверительными интервалами; hash-разбиения — пустые маркеры. Overlap по выборкам не рисуется: на синтетических выборках он равен 1 у всех стратегий по построению (документ с обоими словами лежит в шардах обоих слов), различает их только число шардов |
| `reports/figures/overview_fanout_by_sample.pdf` | то же для среднего числа шардов в покрытии |
| `reports/figures/overview_gain_fanout.pdf` | парная разность среднего числа шардов в покрытии каждой семантической стратегии с её hash-разбиением того же числа шардов, с доверительными интервалами, панель на выборку |
| `reports/figures/overview_quality_vs_cost.pdf` | overlap против трёх видов стоимости (ряд на бюджет из `figures.budgets`) — дублирования, числа шардов в покрытии и объёма опроса — для всех алгоритмов и стратегий; цвет — алгоритм, форма маркера — семейство стратегии, пустой маркер — hash-разбиение, размер — число реплик |
| `reports/figures/overview_budget_curves.pdf` | overlap и объём опроса при каждом бюджете для стратегий `figures.focus_strategies` и их hash-разбиений, все алгоритмы |
| `reports/figures/overview_slices.pdf` | overlap стратегий `figures.focus_strategies` и их hash-разбиений (ряд на бюджет из `figures.budgets`) по срезам отложенных запросов: квартили средней NPMI между словами запроса (Q1 — слабо связанные слова, Q4 — сильно связанные), запросы со словами вне графа, запросы из одного слова; и по числу слов в запросе |
| `reports/figures/overview_ablations.pdf` | тепловая карта абляций: изменение overlap к базовому варианту в процентных пунктах, варианты × алгоритмы, панель на стратегию из `figures.ablation_strategies` |
| `reports/figures/<method>/fanout_ecdf.pdf` | распределение числа шардов в покрытии по стратегиям; значение в точке 1 — доля запросов в один шард |
| `reports/figures/<method>/overlap_by_budget.pdf` | overlap в зависимости от числа опрошенных шардов |
| `reports/figures/<method>/duplication_vs_fanout.pdf` | дублирование документов против среднего числа шардов в покрытии, точка на стратегию; подпись точки — overlap при бюджетах `figures.budgets` |
| `reports/figures/<method>/fanout_by_sample.pdf` | среднее число шардов в покрытии каждой стратегии на каждой выборке запросов |
| `reports/figures/<method>/cluster_sizes.pdf` | распределение размеров кластеров и крупнейшие кластеры |
| `reports/figures/<method>/cluster_wordclouds.pdf` | облака слов крупнейших кластеров |
| `reports/figures/<method>/graph_clusters.pdf` | граф слов с раскраской по кластерам (вершины наибольшей степени) |
| `reports/figures/<method>/tsne.pdf` | t-SNE слов крупнейших кластеров по строкам матрицы смежности |
| `reports/figures/<method>/cluster_heatmap.pdf` | суммарный вес рёбер между крупнейшими кластерами |

## Промежуточные артефакты

| Путь | Стадия | Содержимое |
|---|---|---|
| `data/processed/pairs.parquet` | `extract_pairs` | пары «запрос — пассаж»: `query`, `doc_id` (MD5 текста), `doc_text`, `is_selected` (отмечен асессором) |
| `data/interim/corpus/documents.parquet` | `build_corpus` | `doc_id`; номер строки — номер документа во всех остальных артефактах |
| `data/interim/corpus/vocabulary.parquet` | `build_corpus` | `term`, `df` — слово и число документов с ним, по алфавиту |
| `data/interim/corpus/doc_terms.npz` | `build_corpus` | разреженная матрица «документ × слово» |
| `data/interim/index/` | `build_index` | полный индекс Whoosh и `CORPUS_FINGERPRINT` — отпечаток корпуса, по которому он построен |
| `models/graph.parquet` | `build_graph` | рёбра графа: `src`, `dst`, `count` (число запросов с обоими словами), `weight` |
| `data/interim/queries.parquet` | `sample_queries` | `sample`, `position`, `query` |
| `data/interim/rankings/<sample>.npz` | `rank_queries` | выдача полного индекса для каждого запроса: номера документов по убыванию оценки, не более `evaluation.ranking_depth` |
| `models/clusterings/<method>.parquet` | `cluster` | `term`, `cluster` |
| `models/partitions/<method>/<strategy>.parquet` | `build_partitions` | `term`, `shard`, `rank` (0 — основной шард слова, 1 и далее — реплики) |
| `data/interim/assignments/<method>/<strategy>.npz` | `assign_documents` | разреженная матрица «документ × шард» |
| `data/interim/evaluation/<method>/queries.parquet` | `evaluate` | измерения по каждому запросу, см. ниже |
| `data/interim/evaluation/<method>/partitions.parquet` | `evaluate` | строки `partitions.csv` для одного алгоритма |
| `data/interim/verification/<method>.json` | `verify` | результаты проверок по стратегиям |
| `data/interim/sensitivity/<method>.parquet` | `sensitivity` | строки `sensitivity.csv` для одного алгоритма |
| `data/interim/ablations/<method>.parquet` | `ablations` | строки `ablations.csv` для одного алгоритма |

Измерения по каждому запросу (`queries.parquet`), строка — стратегия, выборка и запрос:

| Столбец | Значение |
|---|---|
| `strategy`, `sample`, `position` | стратегия, выборка и номер запроса в выборке |
| `evaluated` | у запроса есть выдача полного индекса; иначе измерения пустые |
| `terms` | число разных слов запроса после удаления стоп-слов |
| `known_terms` | число слов запроса, которые есть в корпусе |
| `fanout` | число шардов в покрытии |
| `first_shard` | номер шарда, опрашиваемого первым |
| `hash_probes` | число шардов покрытия из hash-пространства |
| `overlap_<b>`, `volume_<b>` | overlap и объём опроса при каждом бюджете |
