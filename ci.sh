#!/usr/bin/env bash
# ci.sh — lib-kernel-ir CI gate for the bupha shell runner.
#
# uv-managed Python project: sync the env (with the `dev` extra for
# pytest), run the test suite.
set -euo pipefail

cd "$(dirname "$0")"

uv sync --extra dev
uv run pytest tests/ -q
