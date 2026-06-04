# Copilot Instructions

## Repository Structure

Each assignment lives in its own directory (e.g., `assignment1-basics/`) with an independent `pyproject.toml`, tests, and `uv.lock`. Always `cd` into the assignment directory before running commands.

## Build & Test Commands

Package manager: **`uv`** (not pip/conda). All commands must be run from within the assignment directory.

```sh
uv run pytest                                          # all tests
uv run pytest tests/test_model.py                      # single test file
uv run pytest tests/test_model.py::test_swiglu -v      # single test
uv run ruff check cs336_basics/ tests/                 # lint
uv run ruff format cs336_basics/ tests/                # format
uv run ty check                                        # type check
bash make_submission.sh                                # run tests (10s timeout) + zip
```

## Architecture (Assignment 1)

- **`cs336_basics/`** — Student implementation package. Each module (tokenizer, model components, optimizer, training) is a separate file. Starts with skeleton code.
- **`tests/adapters.py`** — Adapter layer connecting student implementations to the test suite. Each `run_*` function wraps student code with a fixed interface. Students fill these in to wire up their implementations.
- **`tests/`** — Pre-written test suite using snapshot-based assertions. `tests/_snapshots/` contains `.npz` and `.pkl` reference outputs. `tests/fixtures/` has test data including a reference model state dict.
- **`config/`** — Model configuration JSON (vocab size, context length, dimensions, etc.).

## Key Conventions

- Tensor type annotations use **jaxtyping**: `Float[Tensor, "batch seq d_model"]`, `Int[Tensor, "..."]`, etc.
- Snapshot testing: `numpy_snapshot.assert_match()` compares against stored `.npz` references with configurable `rtol`/`atol`; `snapshot.assert_match()` uses pickle.
- Python ≥3.12 with modern generics syntax (`class Foo[T: ...]`, `def func[A: (np.ndarray, Tensor)]`).
- Ruff: 120-char line length, `F722` ignored (jaxtyping compatibility), `UP` rules enabled.
- Pytest: runs with `-s` (no capture) and `WARNING` log level.
- Data files go in `data/` (gitignored). See each assignment's README for download commands.
