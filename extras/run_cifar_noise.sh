#!/usr/bin/env bash
# CIFAR-10 label-noise experiments (noise package v3). Cats vs Dogs is already done.
# Run from the REPOSITORY ROOT:
#     QUICK=1 bash extras/run_cifar_noise.sh          # ~30 s smoke test in /tmp
#     caffeinate -i bash extras/run_cifar_noise.sh    # full run, resumable
#     CACHE=/path/to/cache bash extras/run_cifar_noise.sh
# Needs cache/cifar10/main/{train,val,test}_{X,y}.npy, metadata.json, split_manifest.json.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="noise-package-v3-cifar"
CACHE="${CACHE:-cache}"; OUT="${OUT:-results_label_noise}"; TAG="${TAG:-mac}"
W="2,4,8,16,32,48,64,96,128,192,256,512,1024"      # single softmax MLP (NumPy)
TW="1,2,4,8,16,32,64,128,256,512"                   # ten one-vs-rest correction trees (PyTorch)
E=300; TE=100; S="17,29,43"
if [[ "${QUICK:-0}" == "1" ]]; then W="8,32"; TW="4"; E=2; TE=2; S="17"; OUT="/tmp/noise_smoke_cifar"; TAG="smoke"; fi
PY="${PYTHON:-}"; if [[ -z "$PY" ]]; then if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi; fi
echo "$VERSION | python: $($PY -c 'import sys;print(sys.executable, sys.version.split()[0])')"
for f in run.py tree_model.py extras/label_noise_double_descent.py extras/label_noise_tree.py; do
  [[ -f "$f" ]] || { echo "ERROR: $f not found. Run from the repository root and unzip the whole package into it."; exit 1; }
done
grep -q -- "--run-id" extras/label_noise_double_descent.py || { echo "ERROR: old extras/label_noise_double_descent.py; replace it with the one in this package."; exit 1; }
grep -q -- '"cifar10"' extras/label_noise_tree.py || { echo "ERROR: old extras/label_noise_tree.py; replace it with the one in this package."; exit 1; }
D="$CACHE/cifar10/main"
for f in train_X.npy train_y.npy val_X.npy val_y.npy test_X.npy test_y.npy metadata.json split_manifest.json; do
  [[ -f "$D/$f" ]] || { echo "ERROR: missing $D/$f. Set CACHE=/path/to/cache (folder containing cifar10/main)."; exit 1; }
done
$PY -c "import numpy, scipy, torch, matplotlib, sklearn; print('numpy', numpy.__version__, '| torch', torch.__version__)" \
  || { echo "ERROR: missing Python packages; activate the project environment."; exit 1; }
N="$PY extras/label_noise_double_descent.py"
COMMON="--dataset cifar10 --cache $CACHE --out $OUT"
log() { echo; echo "=== $* ($(date '+%H:%M:%S'))"; }

log "1. CIFAR-10 correction TREE (10 OVR trees) under 15% noise"
$PY extras/label_noise_tree.py sweep --dataset cifar10 --cache "$CACHE" --out "$OUT" --run-id ${TAG}_tree_cifar_noise15 --widths $TW --seeds $S --epochs $TE
$PY extras/label_noise_tree.py plot  --dataset cifar10 --out "$OUT" --run-id ${TAG}_tree_cifar_noise15
log "2. CIFAR-10 single softmax MLP, 15% noise, width sweep"
$N sweep $COMMON --run-id ${TAG}_cifar_noise15 --noise 0.15 --widths $W --seeds $S --epochs $E
$N plot  $COMMON --run-id ${TAG}_cifar_noise15 --noise 0.15 --seeds $S
log "3. CIFAR-10 matched zero-noise control"
$N sweep $COMMON --run-id ${TAG}_cifar_noise0 --noise 0.0 --widths $W --seeds $S --epochs $E
$N plot  $COMMON --run-id ${TAG}_cifar_noise0 --noise 0.0 --seeds $S
log "done ($VERSION). Zip '$OUT/cifar10' and send it back."
