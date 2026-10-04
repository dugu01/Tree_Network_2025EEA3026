# Tree-Type Neural Networks — 2025EEA3026

Durgesh Singh (2025EEA3026), IIT Delhi — ELL7285 Machine Learning and Optimization, Assignment 4.
Results compiled October 2026.

Repository: https://github.com/dugu01/Tree_Network_2025EEA3026 (public).

## Start here

- `Tree_Network_2025EEA3026_Report.pdf`: 17-page report (including cover page), including exact test counts.
- `report/main.tex`: editable LaTeX source; compile inside `report/`.
- `report/figures/`, `report/tables/`: report assets and numeric CSV summaries.
- `results/`: original experiment evidence, including all sweep points and 48 final test evaluations.
- `results/final_evaluation_protocol.json`: fixed test comparison set.
- `provenance/report_audit.json`: consistency checks performed during report preparation.
- `results_label_noise/`: label-noise experiments for both datasets (root-only width
  sweeps, clean controls, extra noise draws, epoch-wise run, correction trees).
- `provenance/label_noise_audit.json`: completeness checks and key numbers for the noise study.
- `tests/`: 14 original data/model tests plus four label-noise tests
  (`python -m unittest discover -s tests -p "test_label_noise.py"`).

This is a consolidated source/results package. Keep the original training folder
on the Mac: it contains the large image data, feature caches and trained `.pkl`
checkpoints that were not uploaded in the results bundles. Do not overwrite or
delete that original folder. This package can regenerate report assets without
those checkpoints; checkpoint-dependent commands should run in the original folder.

## Main findings

Across optimization seeds 17, 29 and 43:

| Configuration | Training accuracy | Pruned nodes | Pruned test accuracy, mean ± sample SD |
|---|---:|---:|---:|
| CIFAR-10, width 16 | 100% | 11–12 | 88.09% ± 0.09% |
| Cats vs Dogs, width 8 | 100% | 2 | 97.52% ± 0.15% |
| Cats vs Dogs, width 16 | 100% | 2 | 97.85% ± 0.13% |

A 927,008-parameter frozen ImageNet-pretrained trunk is shared by all nodes.
The compact Cats vs Dogs head has 9,251 parameters; the total is 936,259.
These are transfer-learning results. Root-only CIFAR models can generalize better
than the compact trees. With clean labels, the width sweeps show a validation
log-loss peak near interpolation but no peak in classification error.

With 15% training-label noise (validation/test labels clean, three seeds):

| Model | Test error: before peak → peak → widest |
|---|---|
| CIFAR-10 softmax MLP | 14.7% (w8) → 33.8% (w64) → 19.5% (w1024): complete decrease–increase–decrease curve (softmax MLP only) |
| Cats vs Dogs root-only MLP | 3.6% (w1) → 17.7% (w16) → 10.2% (w1024) |
| Cats vs Dogs correction tree | 4.4% (w1) → 15.4% (w32) → 10.5% (w512) |
| CIFAR-10 correction trees (10 OVR) | 13.8% (w1) → 25.4% (w16) → 16.6% (w512) |

Matched clean-label controls have no comparable peak. Under noise, 76–89% of the
children's targets (below interpolation) are flipped labels, so corrections memorize
noise; the original train-preserving pruning keeps them (test error stays close to
the unpruned tree), while validation-only pruning removes them and restores
root-level test error.

## Build the report

With the Python environment active:

```bash
python extras/build_submission.py
python extras/build_noise_report.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The output is `report/main.pdf`. Alternatively, upload the entire `report/` folder
to Overleaf and compile `main.tex`; all generated assets are included.

The repository address is stored in `report/repository.tex` and included in the
compiled PDF. After rebuilding, copy `report/main.pdf` to
`Tree_Network_2025EEA3026_Report.pdf`. The repository is public, so no access
grant is needed. The cover page expects `report/iitd_logo.png`.

`python run.py report` is the original automatic working-report generator. For the
curated final report, use `extras/build_submission.py` and the commands above.
It is safe to regenerate tables: measurements always come from saved JSON files.

## Checkpoints

Trained `.pkl` checkpoints are not committed; they remain in the original
training folder on the experiment Mac. See `CHECKPOINTS.md` for how they could be
distributed. The recorded metrics, confusion matrices and prediction audits in
`results/` document the reported models.

## Label-noise experiment

All label-noise runs were made on the experiment Mac from the verified feature cache
(`cache/<dataset>/main`). 15% of training labels are corrupted with noise seed
1234 (Cats vs Dogs: 2,621 of 17,476 flipped, approximately 15%; CIFAR-10: 6,750 moved to a random other class);
validation and test labels stay clean. These are exploratory test curves; the clean-label
evaluation protocol (`results/final_evaluation_protocol.json`) is separate and unaffected.

| Folder in `results_label_noise/` | What it is |
|---|---|
| `catsdogs/mac_noise15` | root-only MLP width sweep (17 widths × 3 seeds, 300 epochs) + epoch-wise run (width 16, 1000 epochs) |
| `catsdogs/mac_noise0` | matched clean-label control |
| `catsdogs/mac_noise15_ns5678`, `_ns9012` | two further noise draws |
| `catsdogs/mac_tree_noise15` | correction tree: root / tree / pruned (original rule) / pruned (validation-only), with per-run analysis of what the children learn |
| `catsdogs/near_duplicates.json` | near-duplicate training pair found in feature space |
| `cifar10/mac_cifar_noise15`, `mac_cifar_noise0` | single softmax MLP width sweeps, noisy and clean |
| `cifar10/mac_tree_cifar_noise15` | ten one-vs-rest correction trees under noise |

Scripts (in `extras/`): `label_noise_double_descent.py` (NumPy root-only sweeps,
epoch-wise runs), `label_noise_tree.py` (PyTorch, unchanged `tree_model.py`),
`near_duplicate_check.py`, and the runners `run_noise_experiments.sh` (Cats vs Dogs)
and `run_cifar_noise.sh` (CIFAR-10). From the repository root, with the cache present:

```bash
QUICK=1 bash extras/run_noise_experiments.sh   # smoke test in /tmp
bash extras/run_noise_experiments.sh           # Cats vs Dogs
bash extras/run_cifar_noise.sh                 # CIFAR-10
python extras/build_noise_report.py            # report figures/tables + provenance/label_noise_audit.json
```

To regenerate without touching the archived records, pass `TAG=rerun` (new run IDs)
or `OUT=some_other_folder`. Six runs ended with unusually many noisy-train errors
(late training instability without weight decay); they are kept in all averages and
listed in the report and in the audit file.

## Reproduce training in a clean environment

On the M3 Mac, use native arm64 Python. The measured environment is recorded in
`provenance/observed_environment.json`; `requirements.txt` gives supported ranges.
MPS extracts frozen features, while CPU is used for the small cached-feature MLPs.
A CPU-only machine also works, but timings and exact numerical results may differ.

```bash
conda create -n tree-network python=3.11 -y
conda activate tree-network
python -m pip install -r requirements.txt
python run.py doctor
python run.py selftest
```

The commands below use **separate cache/results directories** to preserve archived
results. CIFAR downloads automatically. Cats vs Dogs reuses a `PetImages/Cat` and
`PetImages/Dog` folder under `data/`, or downloads the Microsoft archive if absent.
Keep custom `--data-root` / `--cats-root` paths consistent across later commands.

```bash
caffeinate -i python run.py sweep --dataset cifar10 --profile main --widths 2,4,8,16,32,64,128,256 --seeds 17,29,43 --epochs 100 --tag capacity_clean_v1 --results reproduction_results --cache reproduction_cache
caffeinate -i python run.py sweep --dataset catsdogs --profile main --widths 1,2,4,8,16,32,64,128 --seeds 17,29,43 --epochs 100 --tag capacity_clean_v1 --results reproduction_results --cache reproduction_cache
```

On Ubuntu, omit `caffeinate -i`. For the full fixed finalization workflow, use an
isolated copy without archived results and retain the default `results` / `cache`
directory names, then run `python extras/finish_evaluation.py`. That helper expects
the exact reviewed source and run names, adds pruning/controls/visualizations, and
evaluates the prespecified models. It does not retrain existing models. On the
original Mac, those final evaluations have already completed; no rerun is needed.

The additional output-refit control can be run on an **unfinalized** run:

```bash
python extras/review_controls.py --run-dir reproduction_results/cifar10/main/sweep_capacity_clean_v1_w16_s17
```

It deliberately refuses finalized runs. Do not retune using the archived test
scores. Reproducing the original experimental sequence from scratch should keep
test evaluation until after the fixed model set is chosen.

## Architecture and implementation

- `data_features.py`: official datasets, image validation, exact decoded-pixel
  deduplication, complete conflicting-label-group exclusion, stratified splits,
  frozen MobileNetV3-Small features and verified caches.
- `tree_model.py`: binary MLP correction nodes, balanced initial BCE, constrained
  unweighted logistic readout refitting, empirical acceptance, ten OVR CIFAR roots,
  greedy pruning and exact bounded-correction inference.
- `run.py`: CLI, checkpoint/metric persistence, source provenance and test evaluation.
- `explain.py`: local pre-threshold Grad-CAM, occlusion maps and top validation images.
- `extras/review_controls.py`: root-refit ablation and prediction-level audits.
- `extras/finish_evaluation.py`: final prespecified cached-model workflow.
- `extras/build_submission.py`: JSON-to-table/figure builder; no model fitting.

Head size is the sum of retained MLP parameters and correction scalars. Total size
adds the frozen trunk once. Pruning allows zero training-accuracy loss and at most
0.5 percentage-point validation loss against the original tree, not accumulated
against a moving reference. The observed final pruned models retained the original
validation error counts. No proof of a globally smallest architecture is claimed.

## Provenance and validity

Current experiment-core SHA256:
`f9efc1a849247bef675e6d6cbb653176fcf231a182fe56ab15b76d4469167fd9`.
Earlier CIFAR baseline SHA256:
`c7c707c1e67d42656c3450f1d2b426e82b5bc9fe4942608c3f23d12ed716594a`.
The older core is retained in `provenance/source_v0_1/`; do not copy it over the
current core. Root-level Python files are hashed together, so report helpers live
under `extras/`. Existing run/configuration source mismatches are rejected.

The shared splits are fixed. Repeated seeds are not independent test samples;
means/SDs describe optimization variability. Original absolute file/cache paths in
manifests describe the experiment machine. Dataset images and caches are excluded. Trained checkpoints are unavailable in
this package; see `CHECKPOINTS.md` before claiming full weight reproducibility.

The repository-ready package includes literature links in the PDF and LaTeX.
The original model code is preserved; no method was replaced after inspecting
final test results. Tests and evidence audits support correctness but do not
prove generalization, global optimization, or novel research priority.

