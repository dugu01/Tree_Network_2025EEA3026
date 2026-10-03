# Tree-Type Neural Networks Revisited — 2025EEA3026

Durgesh Singh, IIT Delhi. Results compiled 3 October 2026.

The experiments are complete. The private repository is
https://github.com/dugu01/Tree_Network_Revisited_2025EEA3026.
**Evaluator access must still be granted before submission.**

## Start here

- `Tree_Network_Revisited_Report.pdf`: curated 11-page report, including exact test counts.
- `report/main.tex`: editable LaTeX source; compile inside `report/`.
- `report/figures/`, `report/tables/`: report assets and numeric CSV summaries.
- `results/`: original experiment evidence, including all sweep points and 48 final test evaluations.
- `results/final_evaluation_protocol.json`: fixed test comparison set.
- `provenance/report_audit.json`: consistency checks performed during report preparation.
- `tests/`: 14 data/model tests, including conflicting-label group exclusion.

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
These are transfer-learning results. Classification-error double descent was
not established; repeatable validation-log-loss peaks occur near interpolation.
Root-only CIFAR models can generalize better than the compact trees.

## Build the report

With the Python environment active:

```bash
python extras/build_submission.py
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The output is `report/main.pdf`. Alternatively, upload the entire `report/` folder
to Overleaf and compile `main.tex`; all generated assets are included.

The repository address is stored in `report/repository.tex` and included in the
compiled PDF. After rebuilding, copy `report/main.pdf` to
`Tree_Network_Revisited_Report.pdf`. The URL alone does not grant access: invite
the evaluator to the private repository before submission.

`python run.py report` is the original automatic working-report generator. For the
curated final report, use `extras/build_submission.py` and the commands above.
It is safe to regenerate tables: measurements always come from saved JSON files.

## Reproduce training in a clean environment

On the M3 Mac, use native arm64 Python. The measured environment is recorded in
`provenance/observed_environment.json`; `requirements.txt` gives supported ranges.
MPS extracts frozen features, while CPU is used for the small cached-feature MLPs.
A CPU-only machine also works, but timings and exact numerical results may differ.

```bash
conda create -n tree-revisited python=3.11 -y
conda activate tree-revisited
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
manifests describe the experiment machine. Dataset images, caches and checkpoints
are intentionally excluded from the ZIP and from normal Git commits. To archive
checkpoints privately, copy the original Mac files to suitable private storage
or a release asset and document retrieval; never commit the dataset indiscriminately.

The repository-ready package includes literature links in the PDF and LaTeX.
The original model code is preserved; no method was replaced after inspecting
final test results. Tests and evidence audits support correctness but do not
prove generalization, global optimization, or novel research priority.
