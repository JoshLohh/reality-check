# Reality Check

Reality Check is a lightweight AI-generated image detector designed to stay useful after realistic image transformations such as JPEG compression, blur, resizing, noise, color adjustment, and cropping.

The selected checkpoint is our frozen SID80/CIFAKE20 baseline:

- 80% of the training images come from SID_Set.
- 20% of the training images come from CIFAKE.
- No WildFake images were used during training.
- The pretrained EfficientNet-B0 feature extractor stays frozen.
- Only the final binary classifier is trained.

The model outputs an AI score between 0 and 1. A higher score means the model considers the image more likely to be AI-generated. These scores are useful for ranking and thresholding, but they have not gone through a separate probability-calibration step yet.

## Required Prediction Format

The repository includes the required inference script:

```bash
.venv/bin/python scripts/predict.py \
  --input-dir path/to/images \
  --checkpoint checkpoints/sid80_cifake20/best.pt \
  --output outputs/predictions/sid80_cifake20_preds.json \
  --batch-size 16
```

The output JSON contains one `image_path` and one `pred` score for each supported image:

```json
[
  {
    "image_path": "path/to/image.jpg",
    "pred": 0.873
  }
]
```

## What Is Implemented

The main end-to-end pipeline is working:

- CIFAKE download and manifest creation.
- SID_Set preparation with explicit binary-label filtering.
- Safe WildFake manifest preparation using approved training metadata.
- Deterministic robustness transformations.
- Random online transformations during training.
- A manifest-based image loader.
- ImageNet-compatible image preprocessing.
- An EfficientNet-B0 binary classifier below the 2-billion-parameter limit.
- Guarded training that requires `--allow-training`.
- Best-checkpoint selection using validation ROC AUC.
- Prediction JSON containing `image_path` and `pred`.
- Clean evaluation using ROC AUC, accuracy, precision, recall, and F1.
- Robustness evaluation across clean and transformed conditions.
- CSV, JSON, and Markdown robustness reports.
- Unit tests for data preparation, transforms, loading, prediction, evaluation, model configuration, robustness reporting, and training safety.

The following are still in progress:

- Automated false-positive and false-negative galleries.
- Automated uncertain-prediction analysis.
- A more polished local upload interface.
- Probability calibration.
- A properly content-matched WildFake evaluation set.

## Selected Model

| Component | Selected setting |
| --- | --- |
| Backbone | ImageNet-pretrained EfficientNet-B0 |
| Input | 224 x 224 RGB image |
| Classifier | Dropout 0.2 followed by Linear(1280, 1) |
| Training loss | Binary cross-entropy with logits |
| Inference output | Sigmoid score |
| Total parameters | 4,008,829 |
| Trainable parameters | 1,281 |
| Backbone policy | Frozen feature extractor |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.01 |
| Batch size | 16 |
| Scheduled epochs | 15 |
| Best epoch | 13, selected by validation ROC AUC |

Freezing the backbone made training practical on a MacBook Air M3 with 16 GB of memory. It also gives us a clear baseline: pretrained visual features stay fixed while the final classifier learns the authentic-versus-AI decision.

## Selected Training Data

The selected training and validation manifests are balanced between authentic and AI-generated labels.

### Training Manifest: 10,000 Images

| Source | Authentic (`0`) | AI-generated (`1`) | Total | Share |
| --- | ---: | ---: | ---: | ---: |
| SID_Set | 4,000 | 4,000 | 8,000 | 80% |
| CIFAKE | 1,000 | 1,000 | 2,000 | 20% |
| **Total** | **5,000** | **5,000** | **10,000** | **100%** |

### Validation Manifest: 2,000 Images

| Source | Authentic (`0`) | AI-generated (`1`) | Total | Share |
| --- | ---: | ---: | ---: | ---: |
| SID_Set | 800 | 800 | 1,600 | 80% |
| CIFAKE | 200 | 200 | 400 | 20% |
| **Total** | **1,000** | **1,000** | **2,000** | **100%** |

Our label policy is:

- `0`: authentic image.
- `1`: fully AI-generated image.
- SID_Set label `2`, representing tampered-style examples, is excluded instead of being forced into either binary class.

## Datasets

| Dataset | Current use | Label policy | Important limitation |
| --- | --- | --- | --- |
| CIFAKE | Earlier baseline, 20% of selected mixed training, and separate evaluation | `REAL -> 0`, `FAKE -> 1` | Images are only 32 x 32 and come from a narrow CIFAR-10/Stable-Diffusion setting. |
| SID_Set | 80% of selected mixed training and separate evaluation | authentic `0`, full synthetic `1`, tampered label `2` excluded | Strong SID_Set performance does not guarantee cross-generator performance. |
| WildFake | Diagnostic generalization evaluation only | determined from official metadata | No WildFake images trained the selected checkpoint. The current AFHQ-versus-DDIM slice is content-mismatched. |

Raw images remain under `data/raw/` and are ignored by git.

CSV manifests contain paths, labels, source names, split information, and license notes. They do not contain the image bytes.

## Data-Safety Rules

The data preparation pipeline follows these rules:

- Never use test, holdout, challenge, or demo manifests for training.
- Keep training and validation image paths separate.
- Use only binary labels `0` and `1` for this baseline.
- Use official WildFake training metadata only.
- Exclude WildFake rows containing COCO, DALL-E, DALL-E Advanced, or Advanced indicators.
- Do not use the organizer's COCO val2017 or DALL-E Advanced demonstration subset for training or model selection.
- Keep raw datasets outside git.

The WildFake preparation helper also produces an audit JSON showing which protected or unavailable rows were excluded.

## Model Architecture

The model is based on EfficientNet-B0 from `torchvision`. The original multi-class classifier is replaced with:

```text
EfficientNet-B0 feature extractor
  -> 1280-dimensional feature vector
  -> Dropout(p=0.2)
  -> Linear(1280 -> 1)
  -> raw logit during training
  -> sigmoid score during inference
```

Training uses `BCEWithLogitsLoss`, which combines sigmoid and binary cross-entropy in a numerically stable way. The sigmoid function is applied separately during prediction and evaluation.

## Training Transformations

Transformations are applied in memory. Training does not create transformed image copies on disk.

For each training image:

- There is a 30% chance that it remains clean.
- Otherwise, one robustness transformation is selected randomly.
- After the first transformation, there is a 20% chance that a second transformation is applied.
- The same transformation policy is applied to authentic and AI-generated images.

| Condition | Parameters | Real-world analogue |
| --- | --- | --- |
| JPEG | quality 90, 70, 50, 30 | Social-media and messaging re-encoding |
| Gaussian blur | sigma 0.5, 1.0, 2.0 | Defocus, smoothing, and screenshots |
| Resize | scale 0.5 or 0.25, then restore size | Thumbnails and CDN resizing |
| Gaussian noise | sigma 0.02, 0.05, 0.10 | Sensor noise and poor redistribution |
| Color jitter | brightness, contrast, saturation +/-20% | Filters and automatic enhancement |
| Center crop | retain the central 80% | Reframing and profile-picture cropping |

Validation during training uses clean images. The robustness evaluator later applies each transformation deterministically to the same balanced evaluation selection.

## Selected Training Result

The frozen SID80/CIFAKE20 model ran for all 15 scheduled epochs. Whenever validation ROC AUC improved, `best.pt` was replaced. The best checkpoint came from epoch 13.

| Metric at epoch 13 | Value |
| --- | ---: |
| Training loss | 0.3241 |
| Validation loss | 0.2880 |
| Validation ROC AUC | 0.9640 |
| Validation accuracy | 0.9005 |
| Validation precision | 0.8870 |
| Validation recall | 0.9180 |
| Validation F1 | 0.9022 |

Epochs 14 and 15 remain recorded in `training_history.json`, but neither exceeded epoch 13's validation ROC AUC. Therefore:

```text
checkpoints/sid80_cifake20/best.pt
```

contains the epoch-13 weights.

## Robustness Scoring

The robustness score follows the hackathon formula:

```text
final_score = 0.50 * clean_auc + 0.50 * mean_transformed_auc
```

`mean_transformed_auc` is the average ROC AUC across all non-clean transformation conditions.

## Selected Model Robustness Results

Each recorded evaluation used 1,000 balanced images per condition.

| Evaluation data | Clean AUC | Mean transformed AUC | Final score | Worst condition | Worst AUC |
| --- | ---: | ---: | ---: | --- | ---: |
| SID_Set validation slice | 0.9750 | 0.9706 | 0.9728 | `noise_s0_10` | 0.9571 |
| CIFAKE clean test | 0.8750 | 0.8080 | 0.8415 | `blur_s2_0` | 0.6666 |
| WildFake diagnostic slice | 0.4770 | 0.4146 | 0.4458 | `noise_s0_05` | 0.2913 |

The SID_Set and CIFAKE results show that the model works reasonably well within the two data families represented during training. CIFAKE is more sensitive to strong blur, which is likely related to its original 32 x 32 resolution.

### WildFake Evaluation Warning

The current WildFake diagnostic manifest is not a fair final benchmark.

Its authentic class contains AFHQ photographs of cats, dogs, and wild animals. Its AI-generated class contains DDIM bedroom and CC9K images. Therefore, the evaluation mixes two questions:

1. Is the image AI-generated?
2. Is the image an animal, bedroom, or CIFAR-style image?

The low result is evidence that the model does not currently generalize reliably to this dataset combination, but it should not be reported as a clean WildFake benchmark.

A future evaluation should pair:

- DDIM CC9K images with corresponding CIFAR-style real images.
- DDIM bedroom images with real LSUN bedroom images.

## Error Analysis Summary

False positives mostly appeared on authentic images with characteristics that can look generation-like: polished product photography, HDR-style interiors, food images with text overlays, and distant subjects with low texture. This suggests the model can confuse photographic style, editing, and low-texture regions with AI generation.

False negatives mostly appeared on AI-generated images that reproduced familiar photographic cues well: street scenes, vehicle product images, stop signs, and studio-like object photos. Some examples still had semantic inconsistencies or malformed small text, but the frozen image classifier did not reliably use those clues.

Transformation-induced failures showed a clear pattern:

- On SID_Set, noise sigma 0.10 created 90 new errors among images that were correct when clean; 86 were false negatives.
- On CIFAKE, blur sigma 2.0 created 296 new errors, including 205 false negatives and 91 false positives.
- CIFAKE resizing to 0.25x created 279 new errors.
- CIFAKE noise sigma 0.05 created 285 new errors, of which 284 were false negatives.

Noise tends to suppress the AI score and makes synthetic images look more authentic to the detector. Strong blur and downscaling damage both classes and reduce their separability.

## Experiment History

### CIFAKE Laptop Baseline

The first completed baseline used CIFAKE only and a frozen EfficientNet-B0 backbone.

Recorded clean CIFAKE result on 500 balanced test images:

| Metric | Value |
| --- | ---: |
| ROC AUC | 0.8923 |
| Accuracy | 0.8040 |
| Precision | 0.7676 |
| Recall | 0.8720 |
| F1 | 0.8165 |

Recorded robustness result on 200 balanced images per condition:

| Metric | Value |
| --- | ---: |
| Clean ROC AUC | 0.9318 |
| Mean transformed ROC AUC | 0.8633 |
| Final score | 0.8976 |
| Worst condition | `blur_s2_0` |
| Worst-condition AUC | 0.7034 |

### SID_Set-Only Baseline

A frozen SID_Set model was trained for 15 epochs. Its best validation ROC AUC reached approximately 0.9771, showing that the frozen EfficientNet features separated the SID_Set classes well.

Testing that model on CIFAKE produced much weaker performance, which showed that strong results on one dataset do not guarantee cross-dataset generalization.

### Selected SID80/CIFAKE20 Baseline

The next experiment combined SID_Set and CIFAKE:

- 80% SID_Set.
- 20% CIFAKE.

This became the selected frozen baseline because it retained strong SID_Set performance while incorporating a second image source and generator family.

### Partial-Unfreezing Experiment

A separate experiment continued from the frozen SID80/CIFAKE20 checkpoint and unfroze EfficientNet-B0's final two feature stages plus the classifier.

- Trainable parameters increased from 1,281 to 1,130,673.
- Learning rate was reduced to 0.00001.
- Five additional epochs were run.
- Mixed SID/CIFAKE validation AUC improved to 0.9818.
- Performance on the current content-mismatched WildFake diagnostic slice decreased.

This partial model is retained as an experiment, but it is not the selected baseline described by this README.
Its local config, checkpoint, and history files are not included in this repository snapshot.

## Setup and Installation

Run all commands from the repository root:

```bash
cd "/path/to/reality-check"
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Run the test suite:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Getting CIFAKE

The CIFAKE downloader uses the Kaggle API, so a Kaggle account and API token are required.

Download CIFAKE:

```bash
.venv/bin/python scripts/download_cifake.py
```

Expected folders:

```text
data/raw/cifake/archive/train/REAL
data/raw/cifake/archive/train/FAKE
data/raw/cifake/archive/test/REAL
data/raw/cifake/archive/test/FAKE
```

Create the CIFAKE manifests:

```bash
.venv/bin/python scripts/make_cifake_manifest.py
```

## Getting SID_Set

SID_Set is loaded through Hugging Face. Its official validation split is named `validation`, not `val`.

```bash
.venv/bin/python scripts/prepare_sid_set.py \
  --splits train validation \
  --max-per-label 5000
```

Expected manifests:

```text
data/manifests/sid_set_train.csv
data/manifests/sid_set_val.csv
```

Each manifest contains equal numbers of authentic and fully synthetic images. SID_Set label `2` is excluded.

## Recreating the 80/20 Manifests

Select the SID_Set contribution:

```bash
.venv/bin/python scripts/combine_manifests.py \
  --train-manifests data/manifests/sid_set_train.csv \
  --val-manifests data/manifests/sid_set_val.csv \
  --out-train data/manifests/sid80_train.csv \
  --out-val data/manifests/sid80_val.csv \
  --train-per-source-label 4000 \
  --val-per-source-label 800
```

Select the CIFAKE contribution:

```bash
.venv/bin/python scripts/combine_manifests.py \
  --train-manifests data/manifests/train.csv \
  --val-manifests data/manifests/val.csv \
  --out-train data/manifests/cifake20_train.csv \
  --out-val data/manifests/cifake20_val.csv \
  --train-per-source-label 1000 \
  --val-per-source-label 200
```

Combine both selected subsets:

```bash
.venv/bin/python scripts/combine_manifests.py \
  --train-manifests \
    data/manifests/sid80_train.csv \
    data/manifests/cifake20_train.csv \
  --val-manifests \
    data/manifests/sid80_val.csv \
    data/manifests/cifake20_val.csv \
  --out-train data/manifests/combined_train.csv \
  --out-val data/manifests/combined_val.csv \
  --train-per-source-label 4000 \
  --val-per-source-label 800
```

The final command should report:

```text
10,000 training rows
2,000 validation rows
```

## Reproducing the Selected Frozen Model

First perform a dry run:

```bash
.venv/bin/python scripts/train.py \
  --config configs/sid80_cifake20.yaml \
  --dry-run
```

Check for:

```text
train_rows: 10000
val_rows: 2000
freeze_backbone: true
max_epochs: 15
checkpoint_dir: checkpoints/sid80_cifake20
```

Start training:

```bash
.venv/bin/python scripts/train.py \
  --config configs/sid80_cifake20.yaml \
  --allow-training \
  --device mps
```

Use `--device cuda` on a compatible NVIDIA computer or `--device cpu` if neither MPS nor CUDA is available.

Training writes:

```text
checkpoints/sid80_cifake20/best.pt
checkpoints/sid80_cifake20/training_history.json
```

## Reproducing the Robustness Evaluations

### SID_Set

```bash
.venv/bin/python scripts/evaluate_robustness.py \
  --manifest data/manifests/sid80_val.csv \
  --checkpoint checkpoints/sid80_cifake20/best.pt \
  --out-dir outputs/reports/sid80_cifake20_sidset \
  --limit 1000 \
  --batch-size 16
```

### CIFAKE

```bash
.venv/bin/python scripts/evaluate_robustness.py \
  --manifest data/manifests/test_clean.csv \
  --checkpoint checkpoints/sid80_cifake20/best.pt \
  --out-dir outputs/reports/sid80_cifake20_cifake \
  --limit 1000 \
  --batch-size 16
```

Each output directory contains:

```text
summary.json
robustness_summary.csv
robustness_summary.md
predictions/
```

The current WildFake command is omitted because the authentic and generated content is not yet matched fairly.

## Git and Checkpoint Sharing

The following generated artifacts are intentionally ignored by git:

- Raw and processed images.
- Model checkpoints such as `.pt`, `.pth`, and `.ckpt`.
- Prediction JSON.
- Robustness reports.
- Error-analysis outputs.

This keeps the repository small and avoids committing licensed datasets or large binary files.

Teammates should recreate the datasets locally. The selected checkpoint can be transferred separately as a zip file containing:

- `best.pt`
- `training_history.json`
- `configs/sid80_cifake20.yaml`
- A short note identifying the checkpoint as the frozen SID80/CIFAKE20 epoch-13 model

Raw datasets should not be included in the checkpoint package.

## Repository Structure

```text
reality-check/
  configs/                    # Experiment settings
  data/
    raw/                      # Downloaded images; ignored by git
    processed/                # Generated data; ignored by git
    manifests/                # CSV paths, labels, sources, and split metadata
  docs/                       # Planning, status, and experiment notes
  scripts/                    # Data, training, prediction, and evaluation commands
  src/reality_check/          # Reusable model and pipeline code
  tests/                      # Unit tests
  checkpoints/                # Local weights and histories; ignored by git
  outputs/
    predictions/              # Local prediction JSON; ignored by git
    reports/                  # Local evaluation reports; ignored by git
    error_analysis/           # Planned error-analysis artifacts; ignored by git
```

## Main Limitations

- This is an image-level detector, not a video or audio detector.
- A high score is evidence from the model, not proof that an image is AI-generated.
- The frozen classifier can learn dataset-specific shortcuts instead of universal generation artifacts.
- CIFAKE is low-resolution and differs considerably from normal web images.
- Strong in-distribution ROC AUC does not guarantee cross-generator performance.
- The current WildFake diagnostic comparison is content-mismatched.
- The default 0.5 threshold has not been calibrated for a deployment-specific class balance or cost.
- Automated error galleries and a polished upload interface are not completed.
- Visual error explanations are hypotheses. Grad-CAM or controlled image edits would be needed to confirm which regions drive each decision.

## Next Steps

1. Build a content-matched cross-generator evaluation set without using protected organizer data.
2. Run deeper error analysis on false positives and false negatives from the selected frozen model.
3. Compare training strategies one change at a time while preserving the frozen model as the baseline.
4. Add probability calibration and select a threshold using validation data.
5. Package the selected checkpoint, config, history, robustness tables, and limitations for the team.

## Team Member Contributions

| Team Member | Contribution |
| --- | --- |
| How Wei Chen | UI development and data preparation |
| Josh Loh | Model training and testing |
| Kim Soomin | Model and pipeline design |
| Koh Zhesong Marcus | Robustness evaluation, error analysis, and demo video |
| Sadhana Sivakumar | Robustness evaluation and error analysis |
