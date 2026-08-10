VENV = .venv/bin

.PHONY: setup repro metrics test lint freeze clean article

setup:
	python3 -m venv .venv
	$(VENV)/pip install -r requirements.lock
	$(VENV)/pip install -e . --no-deps

repro:
	. $(VENV)/activate && dvc repro

metrics:
	. $(VENV)/activate && dvc metrics show && dvc metrics diff

test:
	$(VENV)/python tests/smoke_test.py

lint:
	$(VENV)/ruff check sharded_index tests

freeze:
	$(VENV)/pip freeze --exclude-editable > requirements.lock

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -name ".DS_Store" -not -path "./.venv/*" -delete
	find reports -type f \( -name "*.aux" -o -name "*.log" -o -name "*.toc" \
		-o -name "*.out" -o -name "*.mtoc" -o -name "*.synctex.gz" \
		-o -name "*.fls" -o -name "*.fdb_latexmk" \) -delete
	rm -rf sharded_index.egg-info .ipynb_checkpoints notebooks/.ipynb_checkpoints

article:          ## пересобрать таблицы из метрик и обе статьи (RU + EN)
	$(VENV)/python reports/article/build_tables.py
	cd reports/article && latexmk -pdf -interaction=nonstopmode article.tex article_en.tex
