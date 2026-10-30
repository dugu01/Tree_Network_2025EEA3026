# Prespecified continuation protocol (20 October 2026)

This document is an experimental plan, not a statement of measured outcomes.
Scientific status: original baseline inspected; new implementation subject to
correctness tests; real image augmentation and all scientific fits run on the
user's machines. Both CIFAR-10 and cleaned Cats vs Dogs are required.

## Requirement-to-evidence map

| Assignment phrase | Operational answer | Evidence and limits |
|---|---|---|
| Earlier tree-type NN | Same correction targets, small ReLU MLP nodes, nonnegative corrections, convex conditional refit, depth two; binary and 10-OVR classifiers | Preserved baseline and exact no-weighting equivalence test; frozen ImageNet trunk disclosed |
| Augment input samples | Mild pixel transforms BEFORE frozen encoder, three fitted views plus original | Seeded view banks, transform policy, split IDs and hashes; not arbitrary noise on cached features |
| Combine augmentation with overfitting | Fit a finite augmented bank; compare original-fit-label accuracy, all fitted-view accuracy and all-views-correct fraction | Zero classification errors defines interpolation; finite-logit cross entropy need not be exactly zero |
| See double descent | Separate capacity, actual root epoch and whole-tree training-budget curves, with clean/noisy labels | Plot train error with held-out error AND loss; identify initial descent, interpolation deterioration, later descent separately; absence is valid |
| Improvement in test accuracy | Paired no-augmentation/augmentation controls and hard-vote ensembles at full training size | Fixed test protocol, seed spread, confusion matrices, paired image bootstrap; no guaranteed gain |
| Multiple trees | 1,3,5,7 independently trained complete predictors sharing the frozen trunk | In CIFAR each member is ten OVR roots with descendants, not one of ten class detectors |
| Majority voting | Hard-label frequency vote; fixed smallest-class-index tie-break | Multi-class ties recorded even at odd committee size; mean-probability voting is not substituted |
| Bias versus samples and epochs | Crossed 3 training-subset seeds x 2 optimization seeds at 10/25/50% original images and 10/30/100 per-node epochs | Exact finite-replicate Brier decomposition with bias-related proxy; independently trained full-tree budgets distinguished from an SGD path |
| Variance versus samples and epochs | Prediction variance on common held-out examples, plus empirical within-data and between-data-mean components | Original image is sampling unit; augmented views not independent samples; initialization-only variance insufficient |
| Loss landscape | Three row-normalized 2-D slices, fixed data, training/validation losses, decision-flip surfaces | Actual hard child outputs recomputed; biases/edge scalars fixed; slices do not prove global geometry |
| Comment on landscape | Contrast nonconvex representation learning, convex fixed-child readout, discontinuities from hard corrections | Conditional PSD Hessian check only; no ordinary global Hessian across hard switches |
| Innovation | Persistence-weighted child fitting and augmentation-bank diversity across complete-tree ensembles | Matched uniform controls; no claim of previously unpublished research or guaranteed extra marks |
| Logical arguments and evidence | Every conclusion links a numerical table, curve, ablation or mathematical identity | Distinguish facts, hypotheses, experiments, observed results and limitations |
| Brief report, separate continuation | 5–7 page target, small recap and four result sections | New directories and report; earlier assignment not overwritten |

## Fixed data and augmentation rules

Reuse cleaned image IDs and original splits. Verify original cache hashes before
extracting views. Never augment first and randomly split views afterwards. Keep
all variants of an original image together. Standardization is fitted on ORIGINAL
training images in the selected subset only, then fixed for every view and holdout.
No validation/test information fits normalization or branch targets.

CIFAR: padded crop (4 pixels, reflection), horizontal flip, mild brightness /
contrast / saturation 0.1 and hue 0.02, followed by pretrained preprocessing.
Cats/Dogs: crop retaining 80–100% area, aspect ratio 0.9–1.1, horizontal flip,
same mild colour jitter, then official preprocessing. Inspect representative
original/view grids before the long run: label preservation is a plausible design
assumption, not an automatic guarantee for every crop. No vertical flips or strong
rotations. Mixup/CutMix are excluded from the primary experiment because soft or
mixed labels would change the original hard correction-target construction.

Nine cached training views provide three reproducible banks of three views. Each
member also sees the original. Two independent seeded views per original are
never fitted and diagnose invariance on training-image content. These are NOT new
independent test images, and not a measure of robustness to arbitrary corruptions.
Primary test accuracy uses the original deterministic test transform: no test-time
augmentation. Keeping the trunk frozen isolates head effects and is compute-aware;
conclusions do not generalize automatically to end-to-end CNN training.

## Controls, sample size and optimization

- No augmentation: one original view.
- Repeat/exposure control: four copies of the original, same nominal view count
  and epoch budget as augmentation. This separates seeing altered images from
  merely spending more updates. Actual per-node updates and topology are recorded;
  adaptive tree compute need not match exactly even with equal bank size.
- Augmentation: original plus three views, uniform baseline fitting.
- Persistent variant: same bank, seeds, optimizer and readout acceptance rule,
  changed child-fitting weights only.
- Root-only controls retain the same one-versus-rest class formulation as the
  tree. The old noisy single-softmax CIFAR curve is historical context, not a
  paired control for this continuation.
- Clean-label study is primary. Exactly floor(0.15*N) ORIGINAL training labels
  are moved to another class with fixed seed 1234; each view inherits that label.
  Validation/test labels are unchanged. Log both observed-label fit and clean-label
  training accuracy. Perfect fit to noisy labels is not perfect semantic accuracy.
- Early stopping is off. Record failed/interpolation-missing seeds rather than
  deleting them. Increase width or training time only from training/validation
  evidence, in a new tagged plan, before any new test inspection.
- Capacity studies start at 25% of original training data for tractability;
  ensemble comparisons use all original training data. Never pool these as the
  same experiment or imply subset results are full-data results.
- Crossed subsets are stratified and nested within each data seed. Different
  data seeds draw independently from the fixed training pool. At 100% they would
  be identical, so the sample-variance study uses smaller fractions.

A genuine complete double-descent claim needs resolved left descent, peak near
measured interpolation, and right descent, robust enough across seeds to exclude a
single unstable fit. A loss-only peak must not be called classification-error double
descent. A late improvement without a prior deterioration is not epoch-wise double
descent. Tree width changes may change topology; plot parameter counts as well and
use root-only fixed-depth controls to limit this confounder. Densify actual-tree
widths when needed with `--tree-widths` and a new `--tag` plus plan filename.

## Persistence-weighted branch hypothesis

Let g identify an original image, and m_g be the fraction of its available local
views misclassified by the current parent before correction. During CHILD MLP
fitting use extra weight w_g = 0.25 + 0.75*m_g, multiplied by the original balanced
BCE class weight and normalized to mean one. The root fit remains unchanged.
Each view still gets its own correctly recomputed C1/C2/C3/C4 target. No errors
are deleted and weights stay positive. At deeper nodes m_g uses views present in
that local node's training subset, not absent views.

The final output refit remains the earlier unweighted, regularized convex problem;
acceptance still requires a lexicographic improvement of local bank error then
BCE, with the same float32 check. This is not a theorem of better test accuracy.
Compare persistent versus uniform branches at the same bank/seed/width/budget.
A negative outcome is useful: persistent errors can be systematically noisy labels,
so this heuristic could emphasize memorization. Record noise-overlap if extending
this variant to noisy-label runs; initial innovation comparison uses clean labels.

## Ensemble reasoning

Same-bank ensembles isolate optimization diversity. Diverse-bank ensembles vary
both seed and the bank of transformed views. There are only three banks, reused
cyclically across seven members: do not claim seven independent augmentation
samples. Compare mean member accuracy, majority-vote accuracy, disagreement and
double-fault rate. High disagreement alone is insufficient if members are weak.
Report all subset means/ranges for each committee size; do not select the best
subset from test accuracy. Those ranges are not confidence intervals. The canonical
prefix follows numeric seed order, fixed before evaluation. Paired bootstrap
intervals resample EVALUATION IMAGES jointly for both predictors (1000 draws); these
are conditional on fitted models, not training-population or selection uncertainty.

For identically distributed scalar predictors with variance sigma² and common
pairwise correlation rho, the variance of their average is
sigma²[rho+(1-rho)/M]. This motivates measuring correlation/diversity but is not an
exact formula for the classification error of a hard-vote ensemble. The actual
required prediction is the most frequent class; CIFAR uses plurality if no class
has more than half the votes. Lowest class index breaks ties, whose rate is reported.
Count the shared frozen encoder ONCE; add all member heads. End-to-end image
latency and head-only latency are different and should be labelled if measured.

## Empirical bias–variance analysis

Let p_{d,s}(x) be a sigmoid/OVR-softmax class-probability vector for data draw d
and optimization seed s. These are descriptive scores, not claimed calibrated.
For one-hot y and R fitted models, define pbar = mean_r p_r. Exactly:
mean_r ||p_r-y||² = ||pbar-y||² + mean_r ||p_r-pbar||².
Average over the same evaluation images. The first term is mean-predictor error,
a bias-related proxy that includes irreducible label uncertainty and finite-test
limitations, not population squared bias with known P(Y|X). Use denominator R in
this identity; sample-SD bars elsewhere use R-1 and answer a different question.
The within-data plus between-data-mean empirical variance identity is also checked.
With only two initializations per data draw, between-data means contain residual
initialization noise; do not call these unbiased population variance components.

Report these values versus both sample count and per-node budget. Inspect monotonic
or nonmonotonic behaviour, not a presumed textbook bias-down/variance-up story.
Actual root epoch snapshots give a complementary true trajectory. Whole-tree
budgets are repeated constructions, because targets and topology depend on earlier
stages; they are not time points of one jointly trained differentiable tree.

## Loss geometry and comparison limits

The readout with fixed hidden activations/child decisions is logistic regression
with nonnegative correction scalars. Its Hessian is D' W D/n + lambda R, positive
semidefinite; the feasible nonnegative domain is convex. Diagnostic Hessians use
a fixed subsample and are explicitly not the full optimizer's Hessian.

Changing MLP hidden weights is nonconvex. Hard child decisions can jump, so the
full-tree loss is piecewise smooth with possible discontinuities. The plots
recompute hard decisions and show prediction-flip fractions alongside loss.
Use the same normalization, axis radius, sample count and multiple direction pairs
for matched models. A visually flat slice is not a proof of flatness in all
parameter directions, a generalization guarantee, or absence of other minima.
The displayed global root-output BCE is distinct from local class-balanced node
objectives and multiclass test softmax log loss. Use actual fitted labels/views
for training slices, clean validation examples for the other slice. Do not change
augmentations between grid points; that would add Monte Carlo noise to the surface.

## Integrity, schedule and interpretation

20 Oct: install, validate baseline equivalence, extract view banks, inspect views,
real pilot on both machines, estimate runtime. 21 Oct: capacity controls and core
ensembles, begin crossed sample-size study. 22 Oct: complete repetitions, landscape
plots, freeze final selection/evaluation set, evaluate test, assemble brief report.
23 Oct: audit figures against JSON, compile/review PDF, submit with time buffer.
This is a schedule, not a runtime promise. Actual pilot performance determines
whether optional dense/noisy sweeps need reduced scope. Every mandatory question
still needs evidence on both datasets. Use a small real pilot before committing
the machines to a large plan.

The test split has already appeared in the earlier assignment. A new test lock
prevents additional iterative tuning but does not erase historical test exposure.
State that explicitly. Do not reuse old test values as a fresh independent sample.
No new conclusions are populated until measurements exist. Failed runs and absent
phenomena are findings to explain, not reasons to manufacture a positive result.

## Primary literature and how it is used

1. Nakkiran et al., *Deep Double Descent: Where Bigger Models and More Data Hurt*,
   ICLR 2020. https://arxiv.org/abs/1912.02292 — capacity/epoch/sample-size distinctions.
2. Dao et al., *A Kernel Theory of Modern Data Augmentation*, ICML 2019.
   https://arxiv.org/abs/1803.06084 — invariance/variance interpretation; its kernel
   theory is not claimed as a theorem for our hard correction tree.
3. Breiman, *Bagging Predictors*, Machine Learning 1996.
   https://www.stat.berkeley.edu/~breiman/bagging.pdf — aggregation and instability;
   our main ensemble varies seeds/views, so it is not described as bootstrap bagging.
4. Yang et al., *Rethinking Bias-Variance Trade-off for Generalization of Neural
   Networks*, ICML 2020. https://proceedings.mlr.press/v119/yang20j.html — empirical
   bias/variance behaviour may be nonmonotonic.
5. Adlam & Pennington, *Understanding Double Descent Requires a Fine-Grained
   Bias-Variance Decomposition*, NeurIPS 2020. https://arxiv.org/abs/2011.03321 —
   sampling/initialization interactions and ensemble interpretation; theoretical
   results apply to their setting, not automatically our classifier.
6. Li et al., *Visualizing the Loss Landscape of Neural Nets*, NeurIPS 2018.
   https://arxiv.org/abs/1712.09913 — normalized slices and their comparison limits.

Candidate contributions are implementation/experimental ideas, not claims of
research priority. Additional credit is the instructor's judgement; no grade is
promised. Strong evidence, careful controls and a clear explanation are the aim.
