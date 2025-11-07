# Repository Guidelines

## Project Structure & Module Organization
- `vectordb_bench/` – core package
  - `backend/` (clients, datasets, runners, utils)
  - `cli/` (Click commands; entry `vectordbbench.py`)
  - `results/` (generated JSON; git‑ignored)
  - `fig/`, `frontend/`, `config-files/` (sample YAML), `custom/`
- `tests/` – pytest suites (`test_*.py`)
- Tooling: `pyproject.toml`, `Makefile`, `Dockerfile`, `.env.example`, `.devcontainer/`
- Entry points: `init_bench`, `vectordbbench`

## Build, Test, and Development Commands
- Python 3.11+
- Install (dev): `pip install -e '.[test]'`
- Optional clients: `pip install -e '.[all]'` or a specific extra (e.g., `'[pgvector]'`)
- Lint: `make lint` (Black check + Ruff)
- Format: `make format` (Black + Ruff --fix)
- Tests (smoke): `make unittest`
- Tests (all): `pytest -q`
- Run app: `python -m vectordb_bench` or `init_bench`
- CLI help: `vectordbbench --help`

## Coding Style & Naming Conventions
- Formatter: Black, line length 120.
- Linter: Ruff (rules configured in `pyproject.toml`).
- Indentation 4 spaces; prefer type hints.
- Naming: modules/packages `snake_case`; classes `CapWords`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Framework: pytest.
- Location/pattern: `tests/test_*.py` (e.g., `tests/test_dataset.py`).
- Keep tests deterministic; mock external services (S3/OSS, DBs) when feasible.
- Example: `pytest tests/test_dataset.py::TestDataSet::test_download_small -svv`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits where possible: `feat(scope): ...`, `fix: ...`, `docs: ...`.
- Link issues (e.g., `(#613)`) and keep subject imperative and concise.
- PRs must pass CI (`make lint`, `make unittest`), include a clear description, relevant config snippets, and logs/screenshots when applicable.
- Target branches: `main` or `vdbbench_*`.

## Security & Configuration Tips
- Use `.env` (see `.env.example`); never commit secrets.
- Key env vars: `DATASET_LOCAL_DIR` (default `/tmp/vectordb_bench/dataset`), `CONFIG_LOCAL_DIR`, `RESULTS_LOCAL_DIR`, `DEFAULT_DATASET_URL`, `DATASET_SOURCE`, `LOG_LEVEL`.
- Results/logs are git‑ignored (`results/`, `logs/`).
- Place YAML configs in `vectordb_bench/config-files/`; override via `CONFIG_LOCAL_DIR` or CLI `--config-file`. Batch runs: `vectordbbench batchcli --batch-config-file <file>`.
