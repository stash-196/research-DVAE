#!/usr/bin/env bash
# Pilot training for xhro_packet_loss (generates configs if missing, then trains).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

EXPERIMENT="${1:-20260702-XHRO_packet_loss_pilot}"
CONFIG_DIR="config/xhro_packet_loss/generated/${EXPERIMENT}"

if [[ ! -d "$CONFIG_DIR" ]] || [[ -z "$(find "$CONFIG_DIR" -maxdepth 1 -name 'cfg_*.ini' 2>/dev/null | head -1)" ]]; then
  echo "[pilot] Generating configs in $CONFIG_DIR"
  python bin/generate_xhro_packet_loss_configs.py --mode pilot --experiment-name "$EXPERIMENT"
fi

CFG="$(find "$CONFIG_DIR" -maxdepth 1 -name 'cfg_*.ini' | head -1)"
if [[ -z "$CFG" ]]; then
  echo "No config found under $CONFIG_DIR" >&2
  exit 1
fi

echo "[pilot] Training with $CFG"
python bin/train_model.py --cfg "$CFG"