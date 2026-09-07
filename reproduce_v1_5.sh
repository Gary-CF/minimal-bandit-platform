#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$project_root/scripts/benchmark_v1_5.py" "$@"
