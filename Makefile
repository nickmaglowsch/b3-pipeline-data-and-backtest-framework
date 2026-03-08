# B3 Data Pipeline — build targets
# Run `make help` for available commands.

CRATE_DIR := b3_pipeline_rs
PYTHON    := python

export PATH := $(HOME)/.cargo/bin:$(PATH)

.PHONY: help dev-rust build-rust test-rust check-rust test all

help:
	@echo ""
	@echo "Available targets:"
	@echo "  dev-rust    Build Rust extension (debug) and install into active venv"
	@echo "  build-rust  Build optimised release wheel (output: b3_pipeline_rs/target/wheels/)"
	@echo "  test-rust   Run Rust unit tests (cargo test)"
	@echo "  check-rust  Run cargo check only (fast type-check, no full compile)"
	@echo "  test        Run Python test suite (pytest)"
	@echo "  all         dev-rust + test"
	@echo ""

dev-rust:
	cd $(CRATE_DIR) && maturin develop

build-rust:
	cd $(CRATE_DIR) && maturin build --release

test-rust:
	cd $(CRATE_DIR) && cargo test

check-rust:
	cd $(CRATE_DIR) && cargo check

test:
	$(PYTHON) -m pytest tests/ -v

all: dev-rust test
