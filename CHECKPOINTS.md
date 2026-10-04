# Checkpoint recovery and distribution

Status: trained weights remain in the original training folder on the experiment Mac and are not committed; no Release assets yet.
Do not claim the metrics can be reproduced by loading weights until these files
are recovered and their predictions checked. Preserve the original Mac folder.

`run.py` writes each model as `model.pkl` inside its model-kind directory, e.g.
`results/catsdogs/main/sweep_capacity_clean_v1_w8_s17/tree/model.pkl`.
Recover the corresponding root/tree/pruned/control checkpoint files for the
reported runs (seeds 17, 29, 43; Cats vs Dogs widths 8 and 16 and CIFAR width 16),
plus any comparator models used in the report. Retain their existing relative
paths. Only load trusted locally produced pickle files.

After recovering them:

1. Reload them with the original compatible environment and compare training,
   validation and test counts with the archived records. Record SHA256 and sizes.
2. Commit the small verified Cats vs Dogs `model.pkl` head files at their original
   paths; review `.gitignore` and add only the specific verified files.
3. Bundle verified CIFAR checkpoints, their relative paths and SHA256 manifest in
   a ZIP and publish it as a versioned GitHub Release asset. Exclude image data and
   feature caches. Decide repository visibility before publishing any weights.
4. Add the real Release tag, asset filename, hash and download URL to this file.
   Retrieval will then be `gh release download ACTUAL_TAG --repo
   dugu01/Tree_Network_2025EEA3026 --pattern ACTUAL_ASSET.zip`; placeholders here
   are instructions, not an existing release or working download command.

The frozen torchvision backbone and its preprocessing are documented separately;
head-only checkpoint availability does not remove the dependency on those weights.
