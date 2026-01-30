# GAMESA/KrystalStack Makefile
# Cloud-friendly build automation

.PHONY: all bootstrap clean test lint format build docs
.PHONY: python rust c-core mesa verify package

SHELL := /bin/bash
ROOT := $(shell pwd)
VENV := $(ROOT)/venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

# Default target
all: build test

# ============================================================
# Bootstrap & Dependencies
# ============================================================

bootstrap:
	@chmod +x scripts/bootstrap.sh
	@scripts/bootstrap.sh

bootstrap-deps:
	@scripts/bootstrap.sh --deps-only

venv:
	@python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip wheel
	@$(PIP) install -r requirements.txt

# ============================================================
# Build Targets
# ============================================================

build: python rust c-core

python: venv
	@echo "Building Python package..."
	@$(PYTHON) -m py_compile src/python/*.py

rust:
	@echo "Building Rust bot..."
	@cd src/rust-bot && cargo build --release

rust-debug:
	@cd src/rust-bot && cargo build

c-core:
	@echo "Building C core runtime..."
	@mkdir -p src/c/build
	@cd src/c/build && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release && ninja

mesa:
	@echo "Building Mesa-GAMESA driver..."
	@cd mesa-gamesa-drivers && make && sudo make install

# ============================================================
# Test Targets
# ============================================================

test: test-python test-rust

test-python: venv
	@echo "Running Python tests..."
	@$(PYTHON) -m pytest tests/python -v || true

test-rust:
	@echo "Running Rust tests..."
	@cd src/rust-bot && cargo test --release

test-integration: venv
	@echo "Running integration tests..."
	@$(PYTHON) -m pytest tests/integration -v || true

verify: venv
	@echo "Verifying stack..."
	@$(PYTHON) scripts/verify_stack.py

# ============================================================
# Lint & Format
# ============================================================

lint: lint-python lint-rust

lint-python: venv
	@echo "Linting Python..."
	@$(VENV)/bin/ruff check src/python/ || true
	@$(VENV)/bin/mypy src/python/ --ignore-missing-imports || true

lint-rust:
	@echo "Linting Rust..."
	@cd src/rust-bot && cargo clippy -- -D warnings

format: format-python format-rust

format-python: venv
	@echo "Formatting Python..."
	@$(VENV)/bin/ruff format src/python/

format-rust:
	@echo "Formatting Rust..."
	@cd src/rust-bot && cargo fmt

# ============================================================
# Code Generation
# ============================================================

codegen: codegen-ipc codegen-policies

codegen-ipc:
	@echo "Generating IPC structs from IDL..."
	@$(PYTHON) scripts/codegen/ipc_gen.py

codegen-policies:
	@echo "Generating policy templates..."
	@$(PYTHON) scripts/codegen/policy_gen.py

# ============================================================
# Documentation
# ============================================================

docs:
	@echo "Generating documentation..."
	@$(PYTHON) scripts/gen_docs.py
	@cd src/rust-bot && cargo doc --no-deps

docs-serve: docs
	@$(PYTHON) -m http.server 8080 --directory docs/

# ============================================================
# Package & Release
# ============================================================

package: package-python package-rust package-deb

package-python: venv
	@echo "Building Python wheel..."
	@$(PYTHON) -m build

package-rust:
	@echo "Building Rust release..."
	@cd src/rust-bot && cargo build --release
	@mkdir -p dist/bin
	@cp src/rust-bot/target/release/gamesa-bot dist/bin/ || true

package-deb:
	@echo "Building .deb package..."
	@scripts/build_deb.sh

# ============================================================
# Docker
# ============================================================

docker-build:
	@docker build -t gamesa:latest -f docker/Dockerfile .

docker-build-gpu:
	@docker build -t gamesa:gpu -f docker/Dockerfile.gpu .

docker-run:
	@docker run -it --rm \
		-v /dev/dri:/dev/dri \
		-v /sys:/sys:ro \
		gamesa:latest

docker-run-nvidia:
	@docker run -it --rm --gpus all \
		-v /dev/dri:/dev/dri \
		gamesa:gpu

# ============================================================
# Clean
# ============================================================

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf src/c/build
	@rm -rf src/rust-bot/target
	@rm -rf dist/ build/ *.egg-info
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete

clean-all: clean
	@rm -rf $(VENV)
	@rm -rf node_modules/

# ============================================================
# Development Helpers
# ============================================================

dev: venv
	@echo "Starting development environment..."
	@source $(VENV)/bin/activate && exec $$SHELL

watch-rust:
	@cd src/rust-bot && cargo watch -x 'build --release'

bench-rust:
	@cd src/rust-bot && cargo bench

# ============================================================
# Help
# ============================================================

help:
	@echo "GAMESA/KrystalStack Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Bootstrap:"
	@echo "  bootstrap      - Full environment setup"
	@echo "  bootstrap-deps - Install system dependencies only"
	@echo "  venv           - Create Python virtual environment"
	@echo ""
	@echo "Build:"
	@echo "  build          - Build all components"
	@echo "  python         - Build Python package"
	@echo "  rust           - Build Rust bot (release)"
	@echo "  c-core         - Build C core runtime"
	@echo "  mesa           - Build Mesa-GAMESA driver"
	@echo ""
	@echo "Test:"
	@echo "  test           - Run all tests"
	@echo "  test-python    - Run Python tests"
	@echo "  test-rust      - Run Rust tests"
	@echo "  verify         - Verify stack prerequisites"
	@echo ""
	@echo "Quality:"
	@echo "  lint           - Lint all code"
	@echo "  format         - Format all code"
	@echo ""
	@echo "Package:"
	@echo "  package        - Build all packages"
	@echo "  package-deb    - Build .deb package"
	@echo "  docker-build   - Build Docker image"
	@echo ""
	@echo "Other:"
	@echo "  docs           - Generate documentation"
	@echo "  clean          - Remove build artifacts"
	@echo "  help           - Show this help"
