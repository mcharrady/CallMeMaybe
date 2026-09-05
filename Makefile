UV_CACHE_DIR = /goinfre/$(USER)/uv-cache
HF_HOME = /goinfre/$(USER)/hf-home
PY = python3
export UV_CACHE_DIR
export HF_HOME

.PHONY: run install debug lint clean

run:
	uv run $(PY) -m src

install:
	pip install uv
	uv sync
	$(PY) -m pip install flake8 mypy

debug:
	uv run $(PY) -m pdb -m src

lint:
	flake8 src
	mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

clean:
	rm -rf .mypy_cache **/__pycache__