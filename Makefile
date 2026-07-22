.PHONY: lint test test-slow build clean help

help:
	@echo "Available commands:"
	@echo "  make lint       Run ruff linter"
	@echo "  make test       Run fast pytest suite"
	@echo "  make test-slow  Run full pytest suite including slow tests"
	@echo "  make build      Build distribution packages"
	@echo "  make clean      Remove build and cache artifacts"

lint:
	ruff check src scripts tests

test:
	pytest -q -m "not slow"

test-slow:
	pytest -q

build:
	python -m build
	twine check dist/*

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
