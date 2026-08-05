#!/usr/bin/env bash
set -euo pipefail

python app.py --server-name 0.0.0.0 --server-port "${PORT:-7860}"

