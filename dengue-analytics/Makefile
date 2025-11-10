init: install-deps

install-deps:
	@pip install --upgrade pip setuptools wheel
	@pip install --upgrade poetry
	@poetry install --no-root
	@pre-commit install --hook-type commit-msg
	@pre-commit run --all-files

format:
	@poetry run ruff check --select I --fix
	@poetry run ruff format
	@poetry run ruff check --fix --exit-non-zero-on-fix

poetry-export:
	@poetry export --with dev -vv --no-ansi --no-interaction --without-hashes --format requirements.txt --output requirements.txt


run: init
	@poetry run streamlit run src/dashboard/app.py

