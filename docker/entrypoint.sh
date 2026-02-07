#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

cd /app
exec python bot.py
