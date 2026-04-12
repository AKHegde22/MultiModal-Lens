# Publishing Checklist

Use this checklist before tagging a release.

## Release Criteria

- [ ] `ruff check src scripts tests` passes.
- [ ] `pytest -q` passes.
- [ ] `python -m build` succeeds and generates `dist/*`.
- [ ] `twine check dist/*` passes.
- [ ] Preflight check succeeds for intended demo models.
- [ ] `docs/tested_model_matrix.md` is updated.
- [ ] README usage snippets match current CLI/UI options.
- [ ] Version in `pyproject.toml` is updated for the release.
- [ ] Git tag follows `vX.Y.Z` format.

## Recommended Preflight Command

```bash
python scripts/run_preflight.py --family auto --model openai/clip-vit-base-patch32
```

## Recommended Build Command

```bash
pip install -e '.[publish]'
python -m build
twine check dist/*
```

## GitHub Actions

- `CI` workflow validates lint, tests, and package build checks on push/PR.
- `Release Check` runs on tag pushes (`v*`) and publishes build artifacts.

## Related Docs

- `docs/README.md`
- `docs/tested_model_matrix.md`
