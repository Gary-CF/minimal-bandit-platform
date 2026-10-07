#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/reproduce_v2_1.py "$@"
