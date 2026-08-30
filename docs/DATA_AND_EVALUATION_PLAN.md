# Data and Evaluation Plan

This document defines how Reality Check will use data, construct transforms, and report robustness.

## Label Policy

Binary target:

- `0`: authentic image
- `1`: AI-generated image

The prediction score `pred` is the probability of class `1`.

If a dataset includes tampered or partially synthetic labels, we will not silently merge them into the first baseline. We will either exclude them or document a separate experiment.

## Dataset Roles

| Dataset | Planned Role | Notes |
| --- | --- | --- |
| SID_Set | Main candidate for higher-resolution training and validation | Verify label meanings before manifest creation. Use authentic vs full synthetic first. |
| CIFAKE | Pipeline smoke test and quick baseline sanity check | Balanced and easy to process, but only 32x32 images, so it is not enough for the final robustness claim. |
| WildFake | Generalization and cross-generator evaluation | Do not train on the provided validation-demo subset. |

Detailed acquisition instructions live in [DATA_ACQUISITION.md](DATA_ACQUISITION.md).

## Split Strategy

We need splits that test generalization, not memorization.

Planned split files:

- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test_clean.csv`
- `data/manifests/test_holdout.csv`

Rules:

- Keep train, validation, and test images disjoint.
- Split by generator/source when metadata allows.
- Keep real and AIGC class counts balanced or report imbalance.
- Keep the challenge validation-demo subset out of training and hyperparameter tuning.
- Record licenses and dataset origins in manifest metadata.

## Manifest Schema

Required columns:

```text
image_path,label,source_dataset,generator,split,license_notes
```

Optional columns:

```text
width,height,original_format,compression_quality,transform_name,sha256
```

## Training Augmentations

Training transforms should be random and applied online.

Planned policy:

- Resize or crop to the model input size.
- Random JPEG compression with quality sampled from 30, 50, 70, 90.
- Random Gaussian blur with sigma sampled from 0.5, 1.0, 2.0.
- Random downscale and upscale with scale sampled from 0.25 or 0.5.
- Random Gaussian noise with sigma sampled from 0.02, 0.05, 0.10.
- Random brightness, contrast, and saturation jitter up to 20 percent.
- Random center or resized crop around 80 percent.

Important alignment rule:

Apply the same transform policy to real and AIGC images so the model does not learn a data-source shortcut.

## Evaluation Transforms

Evaluation transforms are deterministic and reported individually.

| Condition | Parameter |
| --- | --- |
| clean | no extra transform |
| jpeg_q90 | JPEG quality 90 |
| jpeg_q70 | JPEG quality 70 |
| jpeg_q50 | JPEG quality 50 |
| jpeg_q30 | JPEG quality 30 |
| blur_s0_5 | Gaussian blur sigma 0.5 |
| blur_s1_0 | Gaussian blur sigma 1.0 |
| blur_s2_0 | Gaussian blur sigma 2.0 |
| resize_0_5 | Downscale to 0.5x, then upscale |
| resize_0_25 | Downscale to 0.25x, then upscale |
| noise_s0_02 | Gaussian noise sigma 0.02 |
| noise_s0_05 | Gaussian noise sigma 0.05 |
| noise_s0_10 | Gaussian noise sigma 0.10 |
| color_jitter_20 | Brightness, contrast, saturation plus/minus 20 percent |
| center_crop_80 | Center crop to 80 percent, then resize |

Implementation:

- Transform primitives live in `src/reality_check/transforms.py`.
- `scripts/smoke_test_transforms.py` applies every named condition to a small manifest and writes visual previews.
- The smoke test should use `data/manifests/cifake_sample.csv` first so we validate the pipeline without touching model training.

## Image Loader Contract

The manifest image loader is the bridge between data and the future model.

Input:

- A manifest CSV
- A named transform condition
- A target image size

Output per sample:

- `image_path`
- `label`
- dataset/source metadata
- original image size
- transformed and normalized NumPy array shaped `(3, image_size, image_size)`

The loader lives in `src/reality_check/dataset.py`. It is tested by `tests/test_dataset.py` and can be smoke-tested with:

```bash
python scripts/smoke_test_dataset_loader.py \
  --manifest data/manifests/cifake_sample.csv \
  --transform jpeg_q30 \
  --limit 8
```

## Baseline Model Contract

The first model definition lives in `src/reality_check/model.py`.

Scope:

- Public pretrained `EfficientNet-B0` backbone by default.
- Public `ResNet-50` backup option.
- One binary classifier head.
- One raw output logit per image.
- Sigmoid is applied later during inference to produce `pred`.

Out of scope for this step:

- Training loop
- Checkpoints
- Hyperparameter search
- Evaluation metrics
- Threshold tuning

## Training Guardrail

The training entrypoint exists at `scripts/train.py`, but it is intentionally guarded.

Dry-run command:

```bash
python scripts/train.py --config configs/baseline.yaml --dry-run
```

Actual training later requires both:

- `training.enabled: true` in `configs/baseline.yaml`
- `--allow-training` on the command line

This protects the hackathon workflow from accidentally training before data splits, validation rules, and evaluation discipline are ready.

## Metrics

Primary:

- ROC AUC

Secondary:

- Accuracy
- Precision
- Recall
- F1
- PR AUC if class imbalance appears
- Brier score or expected calibration error

Threshold policy:

- Pick a threshold using the validation set only.
- Do not tune the threshold on final test or validation-demo data.
- Report AUC even when threshold metrics look good.

## Robustness Score

Use this compact score for reporting:

```text
auc_robust = mean(auc over transformed conditions)
final_score = 0.50 * auc_clean + 0.50 * auc_robust
```

Also report the worst-condition AUC because averages can hide brittle behavior.

## Robustness Table Template

| Condition | ROC AUC | Accuracy | AUC Drop vs Clean | Notes |
| --- | ---: | ---: | ---: | --- |
| clean | TBD | TBD | 0.000 | Untransformed test images |
| jpeg_q30 | TBD | TBD | TBD | Heavy social-media compression |
| blur_s2_0 | TBD | TBD | TBD | Strong blur |
| resize_0_25 | TBD | TBD | TBD | Thumbnail style degradation |
| noise_s0_10 | TBD | TBD | TBD | Heavy low-light noise |
| center_crop_80 | TBD | TBD | TBD | Reframing/cropping |
| unseen_generator | TBD | TBD | TBD | Cross-generator holdout |

## Error Analysis Template

False positives:

- Authentic images predicted as AIGC with high confidence.
- Check whether they are compressed, filtered, low resolution, highly saturated, or visually synthetic-looking.

False negatives:

- AIGC images predicted as authentic with high confidence.
- Check whether they come from unseen generators, post-processing, realistic camera-like noise, or strong compression.

Trade-offs:

- Lower threshold catches more AIGC images but increases false positives.
- Higher threshold protects authentic images but misses more synthetic images.
- Strong robustness augmentation can reduce clean accuracy if overused.

## Reproducibility Notes

Record the following for every experiment:

- Git commit
- Config file
- Dataset manifest hashes
- Random seed
- Backbone and pretrained weight source
- Image input size
- Augmentation policy
- Best validation metric
- Final clean and robust metrics
