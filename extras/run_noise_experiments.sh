#!/usr/bin/env bash
# Label-noise experiments for the report (noise package v2).
# Run from the REPOSITORY ROOT (the folder containing run.py and tree_model.py):
#     bash extras/run_noise_experiments.sh                 # full run, resumable
#     QUICK=1 bash extras/run_noise_experiments.sh         # ~20 s smoke test in /tmp
#     CACHE=/path/to/cache bash extras/run_noise_experiments.sh
# CACHE must contain catsdogs/main/{train,val,test}_{X,y}.npy, metadata.json and
# split_manifest.json (the original feature cache). Never writes to results/.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="noise-package-v2"
CACHE="${CACHE:-cache}"
OUT="${OUT:-results_label_noise}"
TAG="${TAG:-mac}"
W="1,2,4,8,12,16,20,24,32,40,48,64,96,128,256,512,1024"
TW="1,2,4,8,12,16,24,32,48,64,128,256,512"
E=300; EE=1000; TE=100; S="17,29,43"
if [[ "${QUICK:-0}" == "1" ]]; then W="4,16"; TW="4"; E=3; EE=5; TE=3; S="17"; OUT="/tmp/noise_smoke"; TAG="smoke"; fi

# ---------- preflight: fail early with a clear message ----------
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
fi
echo "$VERSION | python: $($PY -c 'import sys;print(sys.executable, sys.version.split()[0])')"
for f in run.py tree_model.py extras/label_noise_double_descent.py extras/label_noise_tree.py extras/near_duplicate_check.py; do
  [[ -f "$f" ]] || { echo "ERROR: $f not found. Run from the repository root and unzip the whole noise package into it."; exit 1; }
done
grep -q -- "--run-id" extras/label_noise_double_descent.py || { echo "ERROR: extras/label_noise_double_descent.py is an old version; replace it with the one in this package."; exit 1; }
D="$CACHE/catsdogs/main"
for f in train_X.npy train_y.npy val_X.npy val_y.npy test_X.npy test_y.npy metadata.json split_manifest.json; do
  [[ -f "$D/$f" ]] || { echo "ERROR: missing $D/$f. Set CACHE=/path/to/cache (the folder that contains catsdogs/main)."; exit 1; }
done
$PY - <<'PYEOF' || { echo "ERROR: missing Python packages; activate the project environment (conda activate ...)."; exit 1; }
import numpy, scipy, torch, matplotlib, sklearn
print(f"numpy {numpy.__version__} | torch {torch.__version__} | matplotlib {matplotlib.__version__}")
PYEOF
mkdir -p "$OUT/catsdogs"
N="$PY extras/label_noise_double_descent.py"
COMMON="--dataset catsdogs --cache $CACHE --out $OUT"
log() { echo; echo "=== $* ($(date '+%H:%M:%S'))"; }

log "0. near-duplicate check"
$PY extras/near_duplicate_check.py "$D" > /dev/null
if [[ "$OUT" != "results_label_noise" && -f results_label_noise/catsdogs/near_duplicates.json ]]; then
  mv results_label_noise/catsdogs/near_duplicates.json "$OUT/catsdogs/"
fi

log "1. root-only MLP, 15% noise, width sweep"
$N sweep     $COMMON --run-id ${TAG}_noise15 --noise 0.15 --widths $W --seeds $S --epochs $E
log "2. epoch-wise curve at width 16"
$N epochwise $COMMON --run-id ${TAG}_noise15 --noise 0.15 --width 16 --seeds $S --epochs $EE --every 10
$N plot      $COMMON --run-id ${TAG}_noise15 --noise 0.15 --width 16 --seeds $S

log "3. matched zero-noise control"
$N sweep     $COMMON --run-id ${TAG}_noise0 --noise 0.0 --widths $W --seeds $S --epochs $E
$N plot      $COMMON --run-id ${TAG}_noise0 --noise 0.0 --width 16 --seeds $S

for NS in 5678 9012; do
  log "4. extra noise realisation (noise seed $NS)"
  $N sweep   $COMMON --run-id ${TAG}_noise15_ns$NS --noise 0.15 --noise-seed $NS --widths $W --seeds $S --epochs $E
  $N plot    $COMMON --run-id ${TAG}_noise15_ns$NS --noise 0.15 --width 16 --seeds $S
done

log "5. correction TREE under 15% noise (PyTorch, unchanged tree_model.py)"
$PY extras/label_noise_tree.py sweep --cache "$CACHE" --out "$OUT" --run-id ${TAG}_tree_noise15 --widths $TW --seeds $S --epochs $TE
$PY extras/label_noise_tree.py plot  --out "$OUT" --run-id ${TAG}_tree_noise15

log "done ($VERSION). Zip the folder '$OUT' and send it back."
