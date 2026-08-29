# Build Plan

This document is the working plan for building Reality Check without starting training prematurely.

## Product Definition

Reality Check is a hackathon-scale prototype for robust AI-generated image detection. It should accept a folder of images and produce a calibrated probability for each image indicating the likelihood that it is AI-generated.

The final product is not a production moderation system. It is a convincing proof of concept that shows:

- A clear technical pipeline
- A reliable inference interface
- Robustness under realistic image transformations
- Honest evaluation and error analysis

## User Flow

1. User provides an image directory.
2. The system loads supported image files.
3. Images are resized and normalized.
4. The detector computes an AIGC probability.
5. Results are written to JSON.
6. Evaluation scripts summarize clean vs transformed performance.

Required prediction schema:

```json
[
  {
    "image_path": "example.jpg",
    "pred": 0.42
  }
]
```

## Architecture

```text
Image directory
  -> Image loader
  -> Preprocess
  -> Spatial backbone
  -> Binary classifier head
  -> Probability calibration
  -> JSON predictions
```

Optional upgrade:

```text
Image directory
  -> Image loader
  -> Spatial branch
  -> Frequency branch
  -> Feature fusion
  -> Binary classifier head
  -> Probability calibration
  -> JSON predictions
```

## Baseline Model Choice

Start with one public pretrained model:

- Primary candidate: `EfficientNet-B0`
- Backup candidate: `ResNet-50`
- Stretch candidate: small ViT or ConvNeXt variant

Reasoning:

- These are comfortably below the 2B-parameter rule.
- They are easy to run on limited hackathon compute.
- They let us spend effort on data, augmentation, evaluation, and calibration, which the webinar emphasized as higher leverage than model complexity.

## Milestone 0: Repo Planning

Status: done in this scaffold.

Outputs:

- README with product direction
- Build plan
- Data and evaluation plan
- Baseline config
- Empty repo folders

No training should happen in this milestone.

## Milestone 1: Data Inventory

Goal: know exactly what data we are allowed to use.

Tasks:

- Download approved public/licensed datasets outside git.
- Record dataset source, license, class labels, and intended usage.
- Decide which labels become `real` and `aigc`.
- Exclude the challenge validation-demo subset from training.
- Create a small local sample folder for smoke testing scripts.

Artifacts:

- `docs/DATA_ACQUISITION.md`
- `data/manifests/datasets.md`
- `data/manifests/train.csv`
- `data/manifests/val.csv`
- `data/manifests/test_clean.csv`
- `data/manifests/test_holdout.csv`

Manifest columns:

```text
image_path,label,source_dataset,generator,split,license_notes
```

## Milestone 2: Transform Library

Goal: make real-world redistribution measurable and reproducible.

Tasks:

- Implement stochastic training augmentations.
- Implement deterministic evaluation transforms.
- Name each evaluation condition clearly.
- Add unit tests for transform output size, type, and reproducibility.

Current implementation files:

- `src/reality_check/transforms.py`
- `scripts/smoke_test_transforms.py`
- `tests/test_transforms.py`

## Milestone 2.5: Manifest Image Loader

Goal: make one reliable path from manifest rows to model-ready arrays before any training code exists.

Tasks:

- Read manifest CSVs with `image_path`, `label`, dataset, generator, split, and license metadata.
- Open images safely as RGB.
- Apply one named deterministic transform.
- Resize to the future model input size.
- Normalize with ImageNet mean/std, matching common pretrained backbones.
- Return channel-first arrays shaped `(3, image_size, image_size)`.
- Batch samples into arrays shaped `(batch, 3, image_size, image_size)`.

Current implementation files:

- `src/reality_check/dataset.py`
- `scripts/smoke_test_dataset_loader.py`
- `tests/test_dataset.py`

Evaluation condition names:

```text
clean
jpeg_q90
jpeg_q70
jpeg_q50
jpeg_q30
blur_s0_5
blur_s1_0
blur_s2_0
resize_0_5
resize_0_25
noise_s0_02
noise_s0_05
noise_s0_10
color_jitter_20
center_crop_80
```

## Milestone 3: Baseline Model Definition

Goal: define the architecture without starting model training.

Tasks:

- Use a public pretrained backbone under the 2B-parameter rule.
- Start with `EfficientNet-B0`.
- Keep `ResNet-50` as a backup baseline.
- Replace the original classifier with a binary head that outputs one raw logit.
- Apply sigmoid only during inference so training can later use `BCEWithLogitsLoss`.
- Keep the model definition separate from training code.

Current implementation files:

- `src/reality_check/model.py`
- `scripts/describe_model.py`
- `tests/test_model_spec.py`

No checkpoint is created in this milestone.

## Milestone 4: Baseline Training Code

Goal: implement training, not necessarily win with architecture.

Tasks:

- Load train and validation manifests.
- Balance classes by sampling or class weights.
- Fine-tune the classifier head first.
- Optionally unfreeze the backbone after the head is stable.
- Save the best checkpoint by validation ROC AUC.
- Log config, seed, dataset versions, and metrics.

Training success criteria:

- Training is reproducible from a config.
- The checkpoint can be loaded by the inference script.
- Validation ROC AUC improves over random.
- No validation-demo subset is used.

Current implementation files:

- `src/reality_check/training.py`
- `scripts/train.py`
- `tests/test_training_safety.py`

Current status:

- Training script exists.
- Dry-run planning works.
- Safety checks block accidental training.
- Small guarded optimization loop works.
- First smoke run is recorded in `docs/EXPERIMENT_LOG.md`.

## Milestone 5: Calibrated Inference

Goal: satisfy the required repo deliverable.

Tasks:

- Implement `scripts/predict.py`.
- Accept `--input-dir`, `--checkpoint`, and `--output`.
- Recursively find supported image files.
- Write JSON with `image_path` and `pred`.
- Handle unreadable images gracefully.
- Include a simple smoke test with a tiny image folder.

Current status:

- Mock prediction mode works before a checkpoint exists.
- Real checkpoint mode is wired for a future trained checkpoint.
- Required JSON schema is implemented.

Output contract:

```bash
python scripts/predict.py \
  --input-dir data/raw/cifake/archive/test/REAL \
  --output outputs/predictions/mock_preds.json \
  --mock \
  --limit 10
```

## Milestone 6: Robustness Evaluation

Goal: prove that the model survives more than clean data.

Tasks:

- Implement `scripts/evaluate.py`.
- Evaluate clean and transformed test sets.
- Report ROC AUC for each condition.
- Report accuracy, F1, precision, and recall at the selected threshold.
- Compute the robustness score.
- Save a Markdown-ready summary table.

Current status:

- `scripts/evaluate.py` evaluates a prediction JSON against a manifest.
- It reports ROC AUC, accuracy, precision, recall, and F1.
- `scripts/evaluate_robustness.py` evaluates clean and transformed conditions into a compact table.

Report format:

| Condition | ROC AUC | Accuracy | F1 | Notes |
| --- | ---: | ---: | ---: | --- |
| clean | TBD | TBD | TBD | Baseline test set |
| jpeg_q30 | TBD | TBD | TBD | Heavy compression |
| blur_s2_0 | TBD | TBD | TBD | Strong blur |
| center_crop_80 | TBD | TBD | TBD | Reframed image |

## Milestone 7: Error Analysis

Goal: show judgment, not just scores.

Tasks:

- Save top false positives.
- Save top false negatives.
- Group failures by transformation, dataset, generator, and confidence.
- Identify threshold trade-offs.
- Add representative examples to the Devpost writeup and demo video.

Questions to answer:

- Which transforms hurt most?
- Are false positives mostly heavily edited authentic images?
- Are false negatives mostly unseen generators?
- Does calibration make confidence more honest?

## Milestone 8: Demo And Submission

Goal: make the project easy to judge.

Tasks:

- Record a 2-4 minute demo.
- Show the required command-line prediction flow.
- Show JSON predictions.
- Show robustness table.
- Show two to four error examples.
- Explain limitations and next steps.

Demo script:

1. Problem: synthetic images become harder to detect after reposting.
2. Solution: a detector trained and evaluated for real-world transformations.
3. Pipeline: image folder to calibrated AIGC probabilities.
4. Evidence: clean vs transformed AUC table.
5. Honesty: false positives, false negatives, and trade-offs.

## What We Should Avoid

- Do not train on the provided validation-demo subset.
- Do not optimize only for clean accuracy.
- Do not use proprietary or unlabeled data without documenting assumptions.
- Do not rely on a model above the 2B-parameter rule.
- Do not claim the detector proves truth or intent.
- Do not spend early time on a fancy architecture before the baseline, evaluation, and inference script work.
