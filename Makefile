VENV = .venv/bin

.DEFAULT_GOAL := help
.PHONY: help setup hooks repro results test lint format typecheck check freeze clean article

help:        ## list the targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:       ## create the environment with the pinned dependencies
	python3 -m venv .venv
	$(VENV)/pip install -r requirements.lock
	$(VENV)/pip install -e . --no-deps

hooks:       ## install the pre-commit hooks
	$(VENV)/pre-commit install

repro:       ## run the pipeline: only the stages whose inputs changed
	. $(VENV)/activate && dvc repro

results:     ## print the main results table
	$(VENV)/python -m sharded_index.reporting.summary

test:        ## run the tests
	$(VENV)/python -m pytest

lint:        ## check the code style
	$(VENV)/ruff check sharded_index tests
	$(VENV)/ruff format --check sharded_index tests

format:      ## format the code and apply safe fixes
	$(VENV)/ruff check --fix sharded_index tests
	$(VENV)/ruff format sharded_index tests

typecheck:   ## check the types
	$(VENV)/mypy

check: lint typecheck test  ## run every check

freeze:      ## pin the current environment
	$(VENV)/pip freeze --exclude-editable > requirements.lock

clean:       ## remove caches and LaTeX by-products
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -name ".DS_Store" -not -path "./.venv/*" -delete
	find reports -type f \( -name "*.aux" -o -name "*.log" -o -name "*.toc" \
		-o -name "*.out" -o -name "*.mtoc" -o -name "*.synctex.gz" \
		-o -name "*.fls" -o -name "*.fdb_latexmk" \) -delete
	rm -rf sharded_index.egg-info .mypy_cache .pytest_cache .ruff_cache

article:     ## rebuild the article tables and both PDFs
	$(VENV)/python reports/article/build_tables.py
	cd reports/article && latexmk -pdf -interaction=nonstopmode article.tex article_en.tex
