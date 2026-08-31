# Experiment Log

This file records training/evaluation runs so we can explain results clearly in the hackathon submission.

## Smoke Laptop Run 001

Date: 2026-08-30

Purpose:

- Confirm the full training path works on a Mac laptop without overloading it.
- Use a deliberately small CIFAKE subset before attempting larger training.
- Keep the EfficientNet backbone frozen and train only the binary classifier head.

Command:

```bash
.venv/bin/python scripts/train.py \
  --config configs/smoke_laptop.yaml \
  --allow-training \
  --limit-train 1000 \
  --limit-val 200 \
  --max-epochs 1
```

Setup:

- Dataset: CIFAKE
- Train subset: 1,000 images, balanced real/fake
- Validation subset: 200 images, balanced real/fake
- Model: pretrained EfficientNet-B0
- Frozen backbone: yes
- Trainable parameters: 1,281
- Total parameters: 4,008,829
- Device: MPS
- Epochs: 1
- Batch size: 16

Results:

| Metric | Value |
| --- | ---: |
| Training time | 7.7 seconds |
| Train loss | 0.6612 |
| Validation loss | 0.5910 |
| Validation ROC AUC | 0.8114 |
| Validation accuracy | 0.7350 |
| Validation precision | 0.7701 |
| Validation recall | 0.6700 |
| Validation F1 | 0.7166 |

Local artifacts:

- Checkpoint: `checkpoints/smoke_laptop/best.pt`
- History: `checkpoints/smoke_laptop/training_history.json`

Notes:

- This is not the final model.
- CIFAKE is low-resolution, so this is only a pipeline and feasibility check.
- Next run can safely scale to a larger subset, such as 5,000 training images and 1,000 validation images.

## Baseline 5k Run 002

Date: 2026-08-30

Purpose:

- Train a more meaningful laptop-friendly baseline after the tiny smoke run.
- Keep the run small enough for a MacBook Air M3 with 16GB memory.
- Keep the EfficientNet backbone frozen and train only the binary classifier head.

Command:

```bash
.venv/bin/python scripts/train.py \
  --config configs/baseline_5k.yaml \
  --allow-training \
  --limit-train 5000 \
  --limit-val 1000 \
  --max-epochs 1
```

Setup:

- Dataset: CIFAKE
- Train subset: 5,000 images, balanced real/fake
- Validation subset: 1,000 images, balanced real/fake
- Model: pretrained EfficientNet-B0
- Frozen backbone: yes
- Trainable parameters: 1,281
- Total parameters: 4,008,829
- Device: MPS
- Epochs: 1
- Batch size: 16

Validation results:

| Metric | Value |
| --- | ---: |
| Training time | 30.76 seconds |
| Train loss | 0.5812 |
| Validation loss | 0.4405 |
| Validation ROC AUC | 0.9081 |
| Validation accuracy | 0.8160 |
| Validation precision | 0.7678 |
| Validation recall | 0.9060 |
| Validation F1 | 0.8312 |

Local artifacts:

- Checkpoint: `checkpoints/baseline_5k/best.pt`
- History: `checkpoints/baseline_5k/training_history.json`

Notes:

- This is the current local baseline checkpoint.
- Checkpoints are ignored by git and should be regenerated or uploaded separately only if required by the hackathon.
- CIFAKE is still only a low-resolution starter dataset, so stronger datasets are needed for a better final submission.

## Baseline 5k Clean Test Sample

Date: 2026-08-30

Purpose:

- Check the current baseline on a small balanced clean test subset without heating the laptop with a full 20,000-image run.

Commands:

```bash
.venv/bin/python scripts/predict.py \
  --manifest data/manifests/test_clean.csv \
  --checkpoint checkpoints/baseline_5k/best.pt \
  --output outputs/predictions/baseline_5k_test_500_preds.json \
  --limit 500

.venv/bin/python scripts/evaluate.py \
  --manifest data/manifests/test_clean.csv \
  --predictions outputs/predictions/baseline_5k_test_500_preds.json \
  --out-dir outputs/reports/baseline_5k_test_500
```

Results:

| Metric | Value |
| --- | ---: |
| Count | 500 |
| Positives | 250 |
| Negatives | 250 |
| ROC AUC | 0.8923 |
| Accuracy | 0.8040 |
| Precision | 0.7676 |
| Recall | 0.8720 |
| F1 | 0.8165 |

Notes:

- Generated prediction/report files were cleaned before pushing.
- The commands above reproduce them locally.

## Baseline 5k Robustness Sample

Date: 2026-08-30

Purpose:

- Test whether the model survives the hackathon-required image transformations.
- Use a small balanced subset to keep inference laptop-friendly.

Command:

```bash
.venv/bin/python scripts/evaluate_robustness.py \
  --manifest data/manifests/test_clean.csv \
  --checkpoint checkpoints/baseline_5k/best.pt \
  --out-dir outputs/reports/baseline_5k_robustness_200 \
  --limit 200 \
  --batch-size 16
```

Summary:

| Metric | Value |
| --- | ---: |
| Images per condition | 200 |
| Clean ROC AUC | 0.9318 |
| Mean transformed ROC AUC | 0.8633 |
| Final score estimate | 0.8976 |
| Worst condition | `blur_s2_0` |
| Worst condition ROC AUC | 0.7034 |

Robustness table:

| Condition | ROC AUC | Accuracy | AUC Drop vs Clean |
| --- | ---: | ---: | ---: |
| clean | 0.9318 | 0.8350 | 0.0000 |
| jpeg_q90 | 0.9342 | 0.8450 | -0.0024 |
| jpeg_q70 | 0.9330 | 0.8550 | -0.0012 |
| jpeg_q50 | 0.9020 | 0.8300 | 0.0298 |
| jpeg_q30 | 0.9062 | 0.8050 | 0.0256 |
| blur_s0_5 | 0.9126 | 0.8300 | 0.0192 |
| blur_s1_0 | 0.8196 | 0.7300 | 0.1122 |
| blur_s2_0 | 0.7034 | 0.6350 | 0.2284 |
| resize_0_5 | 0.8593 | 0.7400 | 0.0725 |
| resize_0_25 | 0.7173 | 0.6450 | 0.2145 |
| noise_s0_02 | 0.8758 | 0.7650 | 0.0560 |
| noise_s0_05 | 0.8492 | 0.7400 | 0.0826 |
| noise_s0_10 | 0.8403 | 0.7700 | 0.0915 |
| color_jitter_20 | 0.9312 | 0.8000 | 0.0006 |
| center_crop_80 | 0.9021 | 0.8150 | 0.0297 |

Takeaway:

- JPEG compression, color jitter, and center crop are relatively stable.
- Strong blur and heavy downscale/upscale are the clearest weaknesses.
- The next training improvement should focus on blur and resize robustness before changing model architecture.

## SID_Set Frozen-Backbone Run 003

Date: 2026-08-31

Purpose:

- Move beyond the low-resolution CIFAKE baseline with higher-resolution SID_Set images.
- Train a laptop-manageable EfficientNet-B0 baseline with the same realistic random transformation policy used for robustness training.
- Measure clean and transformed validation performance before attempting full backbone fine-tuning.

Training command:

```bash
.venv/bin/python scripts/train.py \
  --config configs/sid_set.yaml \
  --allow-training \
  --limit-train 5000 \
  --limit-val 1000 \
  --max-epochs 15 \
  --device mps
```

Setup:

- Dataset: SID_Set; labels `0 = real` and `1 = full synthetic`; label `2 = tampered` excluded.
- Training subset: 5,000 balanced images (2,500 real, 2,500 synthetic).
- Validation subset: 1,000 balanced images (500 real, 500 synthetic).
- Model: pretrained EfficientNet-B0.
- Backbone: frozen; 1,281 trainable classifier-head parameters out of 4,008,829 total parameters.
- Device: Apple MPS.
- Batch size: 16.
- Epochs: 15.

Best validation result:

| Metric | Value |
| --- | ---: |
| Best epoch | 15 |
| Validation loss | 0.2550 |
| Validation ROC AUC | 0.9771 |
| Validation accuracy | 0.9060 |
| Validation precision | 0.9060 |
| Validation recall | 0.9060 |
| Validation F1 | 0.9060 |

Robustness evaluation command:

```bash
.venv/bin/python scripts/evaluate_robustness.py \
  --manifest data/manifests/sid_set_val.csv \
  --checkpoint checkpoints/sid_set/best.pt \
  --out-dir outputs/reports/sid_set_e15_robustness_1000 \
  --limit 1000 \
  --batch-size 16
```

Robustness summary:

| Metric | Value |
| --- | ---: |
| Images per condition | 1,000 |
| Clean ROC AUC | 0.9771 |
| Mean transformed ROC AUC | 0.9728 |
| Final score estimate | 0.9749 |
| Worst condition | `noise_s0_10` |
| Worst-condition ROC AUC | 0.9589 |

Notes:

- The model remained strong under JPEG compression, blur, resizing, color jitter, and center crop on this validation sample.
- Heavy Gaussian noise reduced threshold-based accuracy and recall more than ROC AUC, suggesting calibration and threshold selection are worthwhile follow-up work.
- This is a validation-based robustness result, not a final cross-generator test. Evaluate the checkpoint on a separate dataset such as CIFAKE test data or WildFake before making final generalization claims.

Local artifacts:

- Checkpoint: `checkpoints/sid_set/best.pt`
- Training history: `checkpoints/sid_set/training_history.json`
- Robustness report: `outputs/reports/sid_set_e15_robustness_1000/`
