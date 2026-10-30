# Tree-Type Neural Networks — Continuation (Assignment 5) · 2025EEA3026

Continuation of the Assignment 4 correction-tree networks: input augmentation,
majority-vote ensembles, bias–variance across samples and epochs, and loss geometry
on CIFAR-10 and Cats vs Dogs. The Assignment 4 code and report are in the
repository root; this folder is self-contained.

## Start here

- **Report:** [`Tree_Continuation_Report.pdf`](Tree_Continuation_Report.pdf)
  (LaTeX source and figures in [`report/`](report/)).
- **Run records:** `runs/` — request, training metrics and test metrics for all 696
  runs (408 Cats vs Dogs, 288 CIFAR-10), plus test probabilities for the 56
  ensemble members.
- **Analysis outputs:** `analysis/` — capacity, ensemble, bias–variance,
  trajectory and landscape results (JSON, CSV, PNG).
- **Plans and protocol:** `plans/` (frozen job manifests) and `PROTOCOL.md`.

## Main findings

| Question | Result |
|---|---|
| Augmentation + overfitting, double descent | Augmentation coexists with zero fitted training error and lowers clean root-only test error by 0.15–0.50 points at widths 32 and 256; a four-copies exposure control does not. Under 15% label noise it is harmful at small widths. Noisy width sweeps show a peak and recovery, not a complete decrease–increase–decrease curve. |
| Ensembles, majority voting | Seven-member diverse-augmentation ensembles: **97.94%** Cats vs Dogs, **89.91%** CIFAR-10 (+0.13 and +0.70 points over unaugmented). The CIFAR-10 gain holds under a paired bootstrap ([0.30, 1.10]) and score-sum tie-breaking (+0.77). It comes mainly from stronger members and fewer shared errors, not more disagreement. |
| Bias and variance | More original samples lower the bias-related term and variance on CIFAR-10; longer per-node training raises variance and replicate label disagreement. |
| Loss landscape | In normalized 2-D slices, augmented fits rise 2–3× more at radius 0.3 (steeper, not flatter). Only the conditional readout problem is convex. |
| Innovation | Diverse augmentation banks across ensemble members; persistence-weighted child fitting (no convincing benefit, reported as negative). |

All results use a frozen ImageNet-pretrained MobileNetV3-Small trunk (transfer
learning). The test splits are those used in Assignment 4.

## Layout

```
Tree_Continuation_Report.pdf   final report
report/                        LaTeX source + figures
common.py, grouped_model.py    model with grouped views and persistence weighting
prepare_views.py               augmentation (RGB, before the frozen encoder) + features
make_plan.py, experiment.py    frozen manifests and resumable training
finalize.py                    test lock before test evaluation
analyse.py, landscape.py       analysis and loss-landscape slices
bundle_results.py              evidence bundle
selftest.py, validation_*.log  correctness tests (synthetic; not evidence)
start_mac.py                   one-shot bank preparation + pilot on Mac
baseline/                      Assignment 4 data loader and tree model used here
plans/  runs/  analysis/       manifests, per-run records, analysis outputs
banks/<dataset>/               labels, ids, view metadata and augmentation audit only
extras/ensemble_checks.py      recomputes ensemble ties, tie-break, disagreement
```

Not included (too large; kept locally): feature banks (`banks/*/*_X.npy`,
`view_*.npy`, `probe_*.npy`, ~1.7 GB), model checkpoints (`model.pkl`), train/val
prediction files and the evidence ZIPs.

## Verify without retraining

```bash
pip install -r requirements.txt
python extras/ensemble_checks.py
```

This reproduces the Section 4 ensemble table (member and vote accuracy, ties,
score-sum tie-break, disagreement, double fault) and the bootstrap intervals from
the saved test probabilities. Run it from this folder.

## Build the report

```bash
cd report
pdflatex Tree_Continuation_Report.tex
pdflatex Tree_Continuation_Report.tex
```

## Reproduce training

Requires the Assignment 4 feature cache and image data. `DATASET` is `catsdogs`
or `cifar10`.

```bash
pip install -r requirements.txt
python selftest.py
python prepare_views.py --base-cache ../cache/$DATASET/main --data-root ../data --out banks/$DATASET --device mps
python make_plan.py --dataset $DATASET --phase capacity --fraction .25 --epochs 100 --out plans/$DATASET/capacity.json
python experiment.py --bank banks/$DATASET --plan plans/$DATASET/capacity.json --out runs/$DATASET --threads 4
#   repeat make_plan/experiment for: trajectory, ensemble, biasvariance
#   (the exact plans used are in plans/)
python landscape.py --bank banks/$DATASET --run runs/$DATASET/<run_id> --out analysis/$DATASET/landscapes/<run_id>
python finalize.py --bank banks/$DATASET --results runs/$DATASET --plans plans/$DATASET/*.json
python analyse.py --bank banks/$DATASET --results runs/$DATASET --out analysis/$DATASET/final_test --split test
```

Use `--device cpu` if MPS is unavailable. For Cats vs Dogs add
`--cats-root <path to PetImages>` to `prepare_views.py`. Completed fits resume;
changed code, config or bank refuses reuse; a test-locked results folder refuses
new fits. `PROTOCOL.md` gives the definitions, controls and confounders.

## Provenance

`SOURCE_HASHES.json` records the training-code hashes used for the runs;
`runs/*/TEST_LOCK.json` records the checkpoint lock taken before test evaluation.
`extras/ensemble_checks.py` was added after the lock and only reads saved outputs.
Never load untrusted pickle files.
