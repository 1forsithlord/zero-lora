#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/content/zero-lora}
DATA_ROOT=${DATA_ROOT:-/content/data}
POOL=${POOL:-data/reference_pool}
OUT=${OUT:-results/phase5}
DRIVE_DIR=${DRIVE_DIR:-/content/drive/MyDrive/zero-lora-phase5}
TASK=${TASK:-dtd}
RANK=16
EPOCHS=10
LR=${LR:-1e-4}

CONFIGS=(
  "raw 0.0" "raw 0.1" "raw 0.3" "raw 1.0" "raw 3.0"
  "gnsp 1.0" "gnsp 3.0" "gnsp 10.0" "gnsp 30.0"
)
SEEDS=(0 1 2)

cd "$REPO"
[[ -d "$(dirname "$DRIVE_DIR")" ]] || { echo "ERROR: Google Drive not mounted."; exit 1; }
mkdir -p "$DRIVE_DIR" "$OUT"

n=0
for seed in "${SEEDS[@]}"; do
  for cfg in "${CONFIGS[@]}"; do
    read -r loss lam <<<"$cfg"
    n=$((n + 1))
    
    lam_fmt=$(python3 -c "print(f'{$lam:g}')")
    run_id="${TASK}_${loss}_lam${lam_fmt}_r${RANK}_s${seed}"
    
    echo "=== [$n/27] $run_id ==="

    if [[ -f "$DRIVE_DIR/$run_id/summary.json" ]]; then
      echo "already on Drive, skipping training"
    else
      python -u -m zerolora.train --root "$DATA_ROOT" --task "$TASK" --loss "$loss" --lambda "$lam" \
           --rank "$RANK" --epochs "$EPOCHS" --lr "$LR" --seed "$seed" --ref-pool "$POOL" \
           --probe-max-images 2000 --out "$OUT"
      
      [[ -f "$OUT/$run_id/summary.json" ]] || { echo "ERROR: missing summary"; exit 1; }
      rm -rf "${DRIVE_DIR:?}/$run_id"
      cp -r "$OUT/$run_id" "$DRIVE_DIR/"
      echo "copied to $DRIVE_DIR/$run_id"
    fi
  done
done
echo "Phase 5 grid finished. Results in $DRIVE_DIR"