# Reality Check

Robust AI-generated image detection for realistic reposting conditions.

This repo is the planning and implementation home for a hackathon prototype that answers one question: given a directory of images, how likely is each image to be AI-generated after real-world transformations such as JPEG compression, blur, resizing, noise, color adjustment, or cropping?

Current status: prototype baseline is implemented. A small CIFAKE laptop baseline has been trained locally, and generated checkpoints/data are ignored by git.

## Product Goal

Reality Check will be a lightweight image forensics prototype with two final user-facing surfaces:

1. A required command-line inference script that takes an image folder and writes JSON predictions.
2. An optional local demo view that shows predictions, confidence scores, robustness results, and representative errors.

The core output format will be:

```json
[
  {
    "image_path": "path/to/image.jpg",
    "pred": 0.873
  }
]
```

`pred` means the calibrated probability that the image is AIGC-generated. Higher is more likely AI-generated.

## Current Baseline

Implemented pipeline:

```text
CIFAKE manifests
  -> robustness transforms
  -> manifest image loader
  -> EfficientNet-B0 binary classifier
  -> guarded training script
  -> prediction JSON
  -> clean and robustness evaluation
```

Current model:

- Backbone: public pretrained `EfficientNet-B0`
- Head: `Dropout(p=0.2)` plus `Linear(1280 -> 1)`
- Inference output: sigmoid probability, written as `pred`
- Parameter count: 4,008,829
- Smoke/baseline training mode: frozen backbone, 1,281 trainable parameters

Recorded local runs:

| Run | Train / Val | Result | Notes |
| --- | ---: | --- | --- |
| smoke laptop | 1,000 / 200 | validation ROC AUC 0.8114 | pipeline check |
| baseline 5k | 5,000 / 1,000 | validation ROC AUC 0.9081 | current local baseline |
| baseline 5k clean test sample | 500 balanced test images | ROC AUC 0.8923 | user-run clean evaluation |
| baseline 5k robustness sample | 200 balanced test images per condition | clean AUC 0.9318, mean transformed AUC 0.8633 | worst condition: `blur_s2_0` |

See [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) and [docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md).

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
```

Downloaded datasets, generated predictions, evaluation reports, and model checkpoints are intentionally ignored by git.

## Teammate Setup

Raw images and checkpoints are not committed. After cloning the repo, each teammate should recreate local data and checkpoints.

1. Create the environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

2. Download CIFAKE into the expected local path:

```bash
mkdir -p data/raw/cifake
kaggle datasets download \
  -d birdy654/cifake-real-and-ai-generated-synthetic-images \
  -p data/raw/cifake \
  --unzip
```

Expected folders:

```text
data/raw/cifake/archive/train/REAL
data/raw/cifake/archive/train/FAKE
data/raw/cifake/archive/test/REAL
data/raw/cifake/archive/test/FAKE
```

3. Regenerate manifests if needed:

```bash
.venv/bin/python scripts/make_cifake_manifest.py
```

4. Recreate the current local baseline checkpoint if needed:

```bash
.venv/bin/python scripts/train.py \
  --config configs/baseline_5k.yaml \
  --allow-training \
  --limit-train 5000 \
  --limit-val 1000 \
  --max-epochs 1
```

This writes:

```text
checkpoints/baseline_5k/best.pt
```

The checkpoint is ignored by git, so teammates either regenerate it or receive it separately if the hackathon submission process requires sharing model weights.

## Proposed Technical Approach

We will start with a clean, finishable baseline:

1. Preprocess each image with deterministic resizing and normalization.
2. Fine-tune a public pretrained image backbone under the 2B-parameter limit, likely `EfficientNet-B0`, `ResNet-50`, or a small ViT from `timm` or `torchvision`.
3. Add a binary classifier head that predicts real vs AI-generated.
4. Train with realistic random augmentations so the detector learns signals that survive redistribution.
5. Calibrate probabilities on a validation set before final inference.
6. Evaluate clean performance and transformed-image robustness separately.

If the baseline is stable, the main upgrade path is a second frequency branch using FFT or DCT features, fused with the spatial backbone before classification. We will treat that as an optional improvement, not the day-one dependency.

## Robustness Plan

Training will include random versions of the same transformations the challenge asks us to survive:

| Transform | Training/Evaluation Parameters | Real-World Analog |
| --- | --- | --- |
| JPEG compression | quality 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian blur | sigma 0.5, 1.0, 2.0 | Out-of-focus images, screenshot smoothing |
| Resize | scale 0.5 or 0.25, then upscale | Thumbnails, CDN resize |
| Gaussian noise | sigma 0.02, 0.05, 0.10 | Low-light sensor noise |
| Color jitter | brightness, contrast, saturation plus/minus 20 percent | Filter apps, auto-enhance |
| Center crop | crop 80 percent | Profile-picture cropping, reframing |

The key data rule is alignment: we should avoid letting the model learn shortcuts such as "real images are more JPEG-compressed than fake images." Real and AI-generated images should be balanced by source, size, compression, and transformation policy wherever possible.

## Evaluation Strategy

Primary metric: ROC AUC, because it is threshold-free and more stable under class imbalance.

Secondary metrics:

- Accuracy at the chosen validation threshold
- F1 score
- Precision and recall
- Brier score or expected calibration error for confidence quality
- Per-transform AUC drop from clean performance

The compact robustness score we will report:

```text
final_score = 0.50 * auc_clean + 0.50 * auc_robust
```

where `auc_robust` is the mean AUC across the transformed evaluation sets.

## Data Plan

Candidate public datasets from the problem statement:

- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set): large Hugging Face dataset with real, full synthetic, and tampered-style labels. For the first binary baseline, we will use authentic vs full synthetic and document any treatment of tampered images separately.
- [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images): 120k 32x32 images, balanced between real CIFAR-10 images and Stable Diffusion generated images. Useful for smoke tests, but not enough by itself for the final story because it is low-resolution and narrow.
- [WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary): intended for generalization and cross-generator evaluation. The provided validation subset must not be used during training.

All downloaded data stays outside git under `data/raw/` or `data/processed/`.

See [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md) for the exact download order, label mapping, and hackathon-safe data rules.

## Repo Structure

```text
reality-check/
  configs/
    baseline.yaml              # Safe default config; training disabled
    smoke_laptop.yaml          # Small guarded laptop run config
    baseline_5k.yaml           # Current local baseline config
  data/
    raw/                       # Downloaded datasets, ignored by git
    processed/                 # Prepared image folders, ignored by git
    manifests/                 # CSV metadata files; safe to version
  docs/
    BUILD_PLAN.md              # Exact build milestones and responsibilities
    DATA_AND_EVALUATION_PLAN.md # Dataset, split, transform, and metric plan
    EXPERIMENT_LOG.md          # Local run history and metrics
    CURRENT_STATUS.md          # What is done and what remains
  notebooks/
    .gitkeep                   # Future exploration and error analysis notebooks
  scripts/
    .gitkeep                   # Future train/evaluate/predict entrypoints
  src/
    reality_check/
      .gitkeep                 # Future package code
  tests/
    .gitkeep                   # Future unit tests for transforms and inference
  checkpoints/
    .gitkeep                   # Model weights, ignored by git
  outputs/
    predictions/               # JSON prediction files, ignored by git
    reports/                   # Metrics tables and plots, ignored by git
    error_analysis/            # False positive/false negative galleries, ignored by git
```

## Exact Build Steps

1. Finalize the label policy.
   - Label `1` means AI-generated.
   - Label `0` means authentic.
   - Keep validation-demo data separate from all training data.

2. Create data manifests.
   - Build `data/manifests/train.csv`, `val.csv`, `test_clean.csv`, and `test_holdout.csv`.
   - Required columns: `image_path`, `label`, `source_dataset`, `generator`, `split`, `license_notes`.
   - Split by source and generator when metadata allows, so the test set measures generalization.

3. Implement transforms.
   - Training transforms are stochastic.
   - Evaluation transforms are deterministic and named, such as `jpeg_q30`, `blur_s2`, and `crop_80`.
   - Add tests to ensure every transform preserves a valid image and expected output size.

4. Build the baseline detector.
   - Use a public pretrained backbone under 2B parameters.
   - Replace the final head with a binary classifier.
   - Save the best checkpoint by validation ROC AUC.
   - Do not tune on the final validation-demo subset.

5. Add calibrated inference.
   - Implement `scripts/predict.py`.
   - Input: image directory.
   - Output: JSON list with `image_path` and `pred`.
   - Batch images for speed and skip unreadable files with clear warnings.

6. Add evaluation.
   - Implement `scripts/evaluate.py`.
   - Evaluate clean and every transformed condition.
   - Produce `outputs/reports/robustness_summary.csv` and a compact Markdown table for the README or Devpost.

7. Add error analysis.
   - Save top false positives and false negatives.
   - Summarize recurring failure modes: generator type, compression level, semantic category, image resolution, or artifact type.

8. Prepare the demo.
   - Show inference on a small image folder.
   - Show the JSON output.
   - Show clean vs transformed robustness table.
   - Show two to four representative errors and explain trade-offs.

## Useful Commands

Useful commands. Training is guarded and will not start unless the command includes `--allow-training`.

```bash
python scripts/make_cifake_manifest.py
python scripts/smoke_test_transforms.py --manifest data/manifests/cifake_sample.csv --limit 4
python scripts/smoke_test_dataset_loader.py --manifest data/manifests/cifake_sample.csv --transform jpeg_q30 --limit 8
python scripts/describe_model.py
python scripts/train.py --config configs/baseline.yaml --dry-run
python scripts/train.py --config configs/smoke_laptop.yaml --allow-training --limit-train 1000 --limit-val 200 --max-epochs 1
python scripts/predict.py --manifest data/manifests/test_clean.csv --output outputs/predictions/mock_preds.json --mock --limit 200
python scripts/evaluate.py --manifest data/manifests/test_clean.csv --predictions outputs/predictions/mock_preds.json --out-dir outputs/reports/evaluation
python scripts/evaluate_robustness.py --manifest data/manifests/test_clean.csv --checkpoint checkpoints/baseline_5k/best.pt --out-dir outputs/reports/baseline_5k_robustness_200 --limit 200
```

To intentionally start training later, the config must set `training.enabled: true` and the command must include `--allow-training`. The safe default config, `configs/baseline.yaml`, keeps training disabled.

## Deliverables Checklist

- [ ] Public GitHub repo with structured code
- [ ] Inference script that outputs `image_path` and `pred`
- [ ] README with setup, reproduction, limitations, and team contributions
- [ ] Compact robustness evaluation table
- [ ] Error analysis note
- [ ] Devpost written description
- [ ] 2-4 minute public YouTube demo video

## Limitations To State Clearly

- The prototype is image-level only, not video or audio.
- The model will not prove whether an image is fake; it estimates likelihood from learned forensic signals.
- Robustness depends heavily on training data diversity and transformation coverage.
- False positives are especially sensitive for authentic images that have been heavily compressed, filtered, or reposted.
- Cross-generator testing is required because clean in-distribution accuracy can overstate real-world performance.

## Team Contributions

To be filled in before submission.
