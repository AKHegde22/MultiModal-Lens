# Contributing

Thanks for contributing to MultimodalLens.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Local Quality Gates

Run before opening a PR:

```bash
ruff check src scripts tests
pytest -q
```

If making release/publish changes:

```bash
pip install -e '.[publish]'
python -m build
twine check dist/*
```

## PR Guidelines

- Keep changes focused and atomic.
- Add or update tests for behavior changes.
- Update docs when user-facing behavior changes.
- Prefer backward-compatible API changes.

## Commit Message Style

Use clear, imperative messages, for example:

- feat: add model preflight CLI
- fix: handle clip attention fallback
- docs: update publish checklist
