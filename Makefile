# ==== Universal A2A Agent — Makefile ====
# Optimized for idempotent (skippable) installations and efficient development.
# Usage: `make help` (default)


# ---- OS-aware settings ----
OS_UNAME := $(shell uname 2>/dev/null || echo Windows_NT)
VENV ?= .venv
PY ?= python # Using 'python' is more cross-platform

ifeq ($(OS_UNAME),Windows_NT)
	VENV_BIN := $(VENV)/Scripts
else
	VENV_BIN := $(VENV)/bin
endif

PIP = $(VENV_BIN)/pip
UVICORN = $(VENV_BIN)/uvicorn
PYTHON = $(VENV_BIN)/python
CLI = $(VENV_BIN)/a2a

# ---- Core Project Settings ----
PORT ?= 8000
HOST ?= 0.0.0.0
BASE ?= http://localhost:$(PORT)
TEXT ?= ping from Makefile
PID_FILE ?= .uvicorn.pid

# Stamp files track completed installations to prevent redundant work
INSTALL_STAMP = $(VENV)/.install-stamp
DEV_DEPS_STAMP = $(VENV)/.dev-deps-installed

# ---- Docker & Deployment Settings ----
IMAGE ?= universal-a2a-agent:local
IMAGE_REPO ?= your-repo/universal-a2a-agent
IMAGE_TAG ?= 1.2.0
EXTRAS ?= all
PLATFORMS ?= linux/amd64
PUSH ?= false
LATEST ?= false
COMPOSE ?= docker compose
NAMESPACE ?= default
RELEASE ?= a2a
CHART_DIR ?= deploy/helm/universal-a2a-agent

# ---- Internal Helpers ----
# Converts space-separated EXTRAS to comma-separated for pip
empty :=
space := $(empty) $(empty)
comma := ,
EXTRAS_CSV := $(subst $(space),$(comma),$(strip $(EXTRAS)))

# Colors for better readability
C_RESET := \033[0m
C_BLUE := \033[1;34m
C_GREEN := \033[1;32m
C_YELL := \033[1;33m

# ---- Core Targets ----
.DEFAULT_GOAL := help

# PHONY targets are actions, not files. This prevents conflicts with files of the same name.
.PHONY: help wizard wizard-ci install install-all install-extras dev-tools clean dist-clean \
run run-dev start stop restart logs card \
ping ping-cli ping-py ping-a2a ping-rpc ping-openai \
test test-local lint format type-check verify-integrations ci \
docker-build docker-run docker-push docker-buildx-push docker-clean \
compose-up compose-down compose-logs compose-restart \
helm-install helm-upgrade helm-uninstall env

# ==============================================================================
# ✨ HELP & DOCUMENTATION
# ==============================================================================
help:
	@echo ""
	@echo "$(C_BLUE)Universal A2A Agent — Makefile$(C_RESET)"
	@echo "A self-documenting guide to common development tasks."
	@echo ""
	@echo "$(C_YELL)--- Local Development ---$(C_RESET)"
	@echo "  make install         - Create venv and install dependencies (skips if up-to-date)."
	@echo "  make test            - Run tests. Fast and efficient after the first run."
	@echo "  make run             - Start the server in the foreground."
	@echo "  make run-dev         - Start the server with automatic reload on code changes."
	@echo "  make lint            - Check code for style and errors."
	@echo "  make format          - Automatically format code."
	@echo ""
	@echo "$(C_YELL)--- Advanced Installation ---$(C_RESET)"
	@echo "  make install-all     - Install with all optional framework extras."
	@echo "  make install-extras  - Install with specific extras (e.g., 'make install-extras EXTRAS=\"langgraph crewai\"')."
	@echo "  make wizard          - Run an interactive setup script."
	@echo ""
	@echo "$(C_YELL)--- Docker & Containers ---$(C_RESET)"
	@echo "  make docker-build    - Build the local Docker image."
	@echo "  make docker-run      - Run the container locally."
	@echo "  make compose-up      - Start services using Docker Compose."
	@echo "  make compose-down    - Stop services started with Docker Compose."
	@echo ""
	@echo "$(C_YELL)--- Cleanup & Utilities ---$(C_RESET)"
	@echo "  make clean           - Remove temporary files like Python caches."
	@echo "  make dist-clean      - Perform a full cleanup, including the virtual environment."
	@echo "  make env             - Print key Makefile variables for debugging."
	@echo ""

# ==============================================================================
# ⚙️ ENVIRONMENT & INSTALLATION
# ==============================================================================
# Creates the virtual environment only if the pip executable doesn't exist.
$(VENV_BIN)/pip:
	@echo "$(C_YELL)🔧 [venv] Creating virtual environment in $(VENV)...$(C_RESET)"
	@$(PY) -m venv $(VENV)
	@$(VENV_BIN)/pip install --upgrade pip

# Creates the main install stamp file.
# This rule runs ONLY if the stamp file is missing OR if pyproject.toml is newer.
$(INSTALL_STAMP): $(VENV_BIN)/pip pyproject.toml
	@echo "$(C_YELL)📦 [pip] Installing/updating package from pyproject.toml...$(C_RESET)"
	@$(PIP) install -e .
	@touch $@

# The primary, user-facing install target. It's smart and skips unnecessary work.
install: $(INSTALL_STAMP)
	@echo "$(C_GREEN)✅ [install] Package is up to date.$(C_RESET)"

install-all: $(VENV_BIN)/pip pyproject.toml
	@echo "$(C_YELL)📦 [pip] Installing package with [all] extras...$(C_RESET)"
	@$(PIP) install -e .[all]
	@touch $(INSTALL_STAMP)

install-extras: $(VENV_BIN)/pip pyproject.toml
ifeq ($(strip $(EXTRAS)),)
	@echo "$(C_YELL)⚠️ No EXTRAS provided. Example: make install-extras EXTRAS=\"langgraph crewai\"$(C_RESET)"; exit 1;
endif
	@echo "$(C_YELL)📦 [pip] Installing extras: $(EXTRAS)...$(C_RESET)"
	@$(PIP) install -e .[$(EXTRAS_CSV)]
	@touch $(INSTALL_STAMP)

# Installs development tools and creates a stamp file to prevent re-installation.
$(DEV_DEPS_STAMP): $(VENV_BIN)/pip
	@echo "$(C_YELL)🛠️  [pip] Installing development tools (pytest, ruff, etc.)...$(C_RESET)"
	@$(PIP) install pytest ruff black mypy httpx
	@touch $@

# A user-friendly target to ensure development tools are ready.
dev-tools: $(DEV_DEPS_STAMP)
	@echo "$(C_GREEN)✅ [dev] Development tools are ready.$(C_RESET)"

wizard:
	@echo "$(C_YELL)🧙 [wizard] Launching interactive setup…$(C_RESET)"
	@$(PY) scripts/setup_wizard.py

# ==============================================================================
# 🚀 RUNNING THE APPLICATION
# ==============================================================================
run: install
	@echo "$(C_GREEN)▶️  [run] Starting server at http://$(HOST):$(PORT)$(C_RESET)"
	@$(UVICORN) a2a_universal.server:app --host $(HOST) --port $(PORT)

run-dev: install
	@echo "$(C_GREEN)🔄 [run-dev] Starting server with auto-reload at http://$(HOST):$(PORT)$(C_RESET)"
	@$(UVICORN) a2a_universal.server:app --host $(HOST) --port $(PORT) --reload

start: install
	@echo "$(C_GREEN)🚀 [start] Starting server in background. PID -> $(PID_FILE)$(C_RESET)"
	@nohup $(UVICORN) a2a_universal.server:app --host $(HOST) --port $(PORT) >/dev/null 2>&1 & echo $$! > $(PID_FILE)
	@sleep 1
	@echo "PID=$$(cat $(PID_FILE))"

stop:
	@if [ -f $(PID_FILE) ]; then \
		echo "$(C_YELL)🛑 [stop] Killing process with PID $$(cat $(PID_FILE))$(C_RESET)"; \
		kill $$(cat $(PID_FILE)) || true; rm -f $(PID_FILE); \
	else \
		echo "$(C_YELL)🤷 [stop] No PID file found. Server may not be running.$(C_RESET)"; \
	fi

restart: stop start

# ==============================================================================
# 🧪 QUALITY, TESTING & CI
# ==============================================================================
test: install dev-tools
	@echo "$(C_YELL)🧪 [pytest] Running tests...$(C_RESET)"
	@$(PYTHON) -m pytest -q

test-local: install dev-tools start
	@echo "$(C_YELL)🧪 [pytest] Running integration tests against a live local server...$(C_RESET)"
	@$(PYTHON) -m pytest -q || (RC=$$?; $(MAKE) stop; exit $$RC)
	@$(MAKE) stop

lint: dev-tools
	@echo "$(C_YELL)🧹 [ruff] Linting code...$(C_RESET)"
	@$(VENV_BIN)/ruff check src

format: dev-tools
	@echo "$(C_YELL)💅 [black] Formatting code...$(C_RESET)"
	@$(VENV_BIN)/black src examples tests

type-check: dev-tools
	@echo "$(C_YELL)✍️  [mypy] Running type checks...$(C_RESET)"
	@$(VENV_BIN)/mypy src || true

verify-integrations: install
	@echo "$(C_YELL)🔗 [verify] Running simple integration sanity checks...$(C_RESET)"
	@$(PYTHON) scripts/check_integrations.py

ci:
	@echo "$(C_YELL)🤖 [ci] Running CI pipeline: install, test, lint...$(C_RESET)"
	@$(MAKE) install
	@$(MAKE) test
	@$(MAKE) lint

# ==============================================================================
# 🐳 DOCKER & COMPOSE
# ==============================================================================
docker-build:
	@echo "$(C_YELL)🐳 [docker] Building image: $(IMAGE)...$(C_RESET)"
	@docker build -t $(IMAGE) .

docker-run:
	@echo "$(C_YELL)🐳 [docker] Running container $(IMAGE) on port $(PORT)...$(C_RESET)"
	@docker run --rm -p $(PORT):8000 --env-file .env $(IMAGE)

docker-push:
	@echo "$(C_YELL)🐳 [docker] Pushing image: $(IMAGE)...$(C_RESET)"
	@docker push $(IMAGE)

docker-buildx-push:
	@echo "$(C_YELL)🏗️  [buildx] Building multi-arch images. Platforms=$(PLATFORMS), Push=$(PUSH)$(C_RESET)"
	@IMAGE_REPO=$(IMAGE_REPO) IMAGE_TAG=$(IMAGE_TAG) EXTRAS=$(EXTRAS) PLATFORMS=$(PLATFORMS) PUSH=$(PUSH) LATEST=$(LATEST) ./scripts/build-containers.sh

compose-up:
	@echo "$(C_YELL) composestarting services...$(C_RESET)"
	@IMAGE_REPO=$(IMAGE_REPO) IMAGE_TAG=$(IMAGE_TAG) EXTRAS=$(EXTRAS) HOST_PORT=$(PORT) PORT=$(PORT) $(COMPOSE) up -d --build

compose-down:
	@echo "$(C_YELL) composeStopping services...$(C_RESET)"
	@$(COMPOSE) down

compose-logs:
	@echo "$(C_YELL) composeTailing logs...$(C_RESET)"
	@$(COMPOSE) logs -f --tail=200 a2a-agent

compose-restart:
	@echo "$(C_YELL) composeRestarting services...$(C_RESET)"
	@$(COMPOSE) restart a2a-agent

# ==============================================================================
#  HELM DEPLOYMENT
# ==============================================================================
helm-install:
	@echo "$(C_YELL) [helm] Installing/upgrading release '$(RELEASE)' in namespace '$(NAMESPACE)'...$(C_RESET)"
	@helm upgrade --install $(RELEASE) $(CHART_DIR) -n $(NAMESPACE) --create-namespace

helm-upgrade: helm-install

helm-uninstall:
	@echo "$(C_YELL) [helm] Uninstalling release '$(RELEASE)' from namespace '$(NAMESPACE)'...$(C_RESET)"
	@helm uninstall $(RELEASE) -n $(NAMESPACE)

# ==============================================================================
# 🧹 CLEANUP & UTILITIES
# ==============================================================================
clean:
	@echo "$(C_YELL)🗑️  [clean] Removing Python cache files...$(C_RESET)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache .mypy_cache

dist-clean: clean
	@echo "$(C_YELL)💣 [dist-clean] Removing all build artifacts and the virtual environment...$(C_RESET)"
	@rm -rf dist build *.egg-info $(VENV)

env:
	@echo "$(C_BLUE)--- Makefile Variables ---$(C_RESET)"
	@echo "VENV      = $(VENV)"
	@echo "PYTHON    = $(PYTHON)"
	@echo "HOST      = $(HOST)"
	@echo "PORT      = $(PORT)"
	@echo "IMAGE     = $(IMAGE)"
	@echo "EXTRAS    = $(EXTRAS)"
	@echo "---"