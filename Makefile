.PHONY: help ensure-uv sync test docs docs-build docs-deploy build clean

# Resolved once at parse time: use uv from PATH if present, otherwise the
# path the official installer places it at (installed on demand by ensure-uv).
UV := $(shell command -v uv 2>/dev/null)
ifeq ($(UV),)
UV := $(HOME)/.local/bin/uv
endif

help:
	@echo "Available targets:"
	@echo "  sync         Install/sync dependencies (incl. dev group)"
	@echo "  test         Run the test suite"
	@echo "  docs         Serve docs locally with live reload"
	@echo "  docs-build   Build the docs site (strict mode)"
	@echo "  docs-deploy  Publish docs to the gh-pages branch"
	@echo "  build        Build the package distribution"
	@echo "  clean        Remove build/test/docs artifacts"

ensure-uv:
	@if [ ! -x "$(UV)" ]; then \
		printf "uv not found. Install it to ~/.local/bin now? [Y/n] "; \
		read reply; \
		case "$$reply" in \
			[nN]*) echo "uv is required; aborting."; exit 1 ;; \
			*) curl -LsSf https://astral.sh/uv/install.sh | sh ;; \
		esac; \
	fi

sync: ensure-uv
	$(UV) sync --all-groups

test: ensure-uv
	$(UV) run --no-sync pytest

docs: ensure-uv
	$(UV) run --no-sync --group docs mkdocs serve

docs-build: ensure-uv
	$(UV) run --no-sync --group docs mkdocs build --strict

docs-deploy: ensure-uv
	$(UV) run --no-sync --group docs mkdocs gh-deploy --force

build: ensure-uv
	$(UV) build

clean:
	rm -rf dist build site .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
