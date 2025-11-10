# --- Project knobs -----------------------------------------------------------
SHIP_SRC       ?= .
SHIP_OUT       ?= ./dist/web
SHIP_CONFIG    ?= ./ship.toml
PYTHON         ?= python3
VENV_DIR       ?= .venv

# --- Phony targets -----------------------------------------------------------
.PHONY: help setup clean verify build-web ship-web ship-web-dry version

help:
	@echo "Targets:"
	@echo "  setup              Create venv and install ship CLI deps"
	@echo "  clean              Remove build artifacts"
	@echo "  verify             Sanity-check repo and config"
	@echo "  build-web          Prepare web artifacts (no copy)"
	@echo "  ship-web           Create runnable web copy to $(SHIP_OUT)"
	@echo "  ship-web-dry       Dry-run (no writes)"
	@echo "  version            Show detected version"

setup:
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	@$(VENV_DIR)/bin/pip -q install --upgrade pip
	@$(VENV_DIR)/bin/pip -q install -r tools/requirements-ship.txt
	@echo "✅ setup complete"

clean:
	@rm -rf $(SHIP_OUT)
	@echo "🧹 cleaned $(SHIP_OUT)"

verify:
	@test -f $(SHIP_CONFIG) || (echo "Missing $(SHIP_CONFIG)"; exit 1)
	@test -f tools/ship_web.py || (echo "Missing tools/ship_web.py"; exit 1)
	@echo "🔎 verifying source…"
	@test -d $(SHIP_SRC) || (echo "Missing SHIP_SRC=$(SHIP_SRC)"; exit 1)
	@echo "✅ verify ok"

build-web: verify
	@echo "🧱 (placeholder) build steps for front-end bundling etc."

ship-web: build-web
	@$(VENV_DIR)/bin/python tools/ship_web.py \
		--src "$(SHIP_SRC)" \
		--out "$(SHIP_OUT)" \
		--config "$(SHIP_CONFIG)"
	@echo "🚀 shipped to $(SHIP_OUT)"

ship-web-dry: verify
	@$(VENV_DIR)/bin/python tools/ship_web.py \
		--src "$(SHIP_SRC)" \
		--out "$(SHIP_OUT)" \
		--config "$(SHIP_CONFIG)" \
		--dry-run --verbose

version:
	@$(VENV_DIR)/bin/python tools/ship_web.py --version

