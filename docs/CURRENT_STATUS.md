# Current Status

This repo is ready to push as a prototype baseline, with local datasets/checkpoints ignored by git.

## Implemented

- CIFAKE manifest creation with `REAL -> 0` and `FAKE -> 1`.
- Robustness transforms for JPEG, blur, resize, noise, color jitter, and center crop.
- Manifest image loader that produces normalized `(3, 224, 224)` arrays.
- EfficientNet-B0 binary classifier definition under the 2B-parameter rule.
- Guarded training script with `--allow-training`.
- Prediction script that writes the required `image_path` and `pred` JSON format.
- Evaluation script for ROC AUC, accuracy, precision, recall, and F1.
- Robustness evaluation script that creates clean-vs-transformed metrics.
- Unit tests for transforms, data loading, model spec, prediction, evaluation, robustness script, and training safety.

## Current Local Model

The current local checkpoint is:

```text
checkpoints/baseline_5k/best.pt
```

It was trained with:

- CIFAKE
- 5,000 balanced training images
- 1,000 balanced validation images
- 1 epoch
- frozen EfficientNet-B0 backbone
- 1,281 trainable classifier-head parameters

Validation ROC AUC:

```text
0.9081
```

This checkpoint is ignored by git.

## Current Results

Clean test sample, 500 balanced images:

```text
ROC AUC: 0.8923
Accuracy: 0.8040
F1: 0.8165
```

Robustness sample, 200 balanced images per condition:

```text
Clean AUC: 0.9318
Mean transformed AUC: 0.8633
Final score estimate: 0.8976
Worst condition: blur_s2_0, AUC 0.7034
```

## Local Files Not Intended For Git

These are intentionally ignored:

- `.venv/`
- `data/raw/cifake/`
- `checkpoints/smoke_laptop/`
- `checkpoints/baseline_5k/`
- `outputs/predictions/`
- `outputs/reports/`

## Recommended Next Steps

1. Add SID_Set manifests so the project is not only based on 32x32 CIFAKE images.
2. Add WildFake as a generalization/holdout dataset.
3. Improve training augmentation based on current weaknesses: strong blur and heavy resize.
4. Run a larger but still laptop-safe baseline, such as 10,000 train and 2,000 validation images.
5. Add error analysis for false positives and false negatives.
6. Prepare the Devpost writeup and demo script.

## Before Pushing

Run:

```bash
.venv/bin/python -m unittest discover -s tests
git status --short --ignored
```

Expected:

- Tests pass.
- Raw data, checkpoints, virtualenv, and generated outputs appear ignored.
- Source, docs, configs, scripts, tests, and manifests are ready to commit.
