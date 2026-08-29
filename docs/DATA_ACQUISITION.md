# Data Acquisition

This document explains how to get data for Reality Check while staying inside the hackathon rules.

Current status: no data has been downloaded, no manifests have been created, and no training has started.

## Hackathon Rules We Must Follow

- Use only public or properly licensed datasets.
- Do not use proprietary, private, platform, or production data.
- Do not train on the validation-demo subset mentioned in the problem statement.
- Do not use test labels for training, threshold tuning, calibration, or model selection.
- Generated augmented samples are allowed only if they come from approved training data and the generation process is reproducible.
- Use public/open-source pretrained backbones only, under the 2B-parameter rule.
- Document every dataset source, license, split, and label mapping.

## Are The Datasets Labelled?

Yes, the three suggested datasets are labelled, but not in exactly the same format.

| Dataset | Label Format | Our Binary Mapping | How We Use It |
| --- | --- | --- | --- |
| SID_Set | Numeric labels: `0`, `1`, `2` | `0 -> authentic`, `1 -> AIGC`; exclude `2` at first | Main higher-resolution training/validation candidate |
| CIFAKE | Folder/class labels: `REAL`, `FAKE` | `REAL -> 0`, `FAKE -> 1` | Fast smoke test and quick baseline sanity check |
| WildFake | Folder or metadata categories, expected real vs generated hierarchy | `Real -> 0`, generated categories -> `1` | Holdout and cross-generator evaluation |

SID_Set label policy:

- `0`: real/authentic
- `1`: full synthetic
- `2`: tampered/manipulated

For the first hackathon baseline, use only `0` and `1`. Keep `2` out of the binary classifier until we explicitly decide how to handle tampered images.

CIFAKE label policy:

- `REAL`: authentic CIFAR-10 image
- `FAKE`: Stable Diffusion generated image

WildFake label policy:

- Real images are non-AIGC.
- GAN, diffusion, or other generated categories are AIGC.
- Verify the exact downloaded folder and annotation structure before creating manifests.

## Recommended Data Order

Start small, then scale:

1. CIFAKE
   - Best first dataset because it is simple and balanced.
   - Use it to test data loading, labels, transforms, and manifest creation.
   - Do not rely on it alone for the final story because the images are only 32x32.

2. SID_Set
   - Best candidate for the main prototype dataset.
   - Use `real` vs `full_synthetic` for the first binary version.
   - Exclude `tampered` initially.

3. WildFake
   - Best for generalization and cross-generator evaluation.
   - Use as holdout/evaluation where possible.
   - Keep the challenge validation-demo subset fully isolated.

## Where Data Should Live

Data should stay out of git.

```text
data/
  raw/
    cifake/
    sid_set/
    wildfake/
    challenge_validation_demo/   # never used for training
  processed/
    sample_images/
  manifests/
    datasets.md
    train.csv
    val.csv
    test_clean.csv
    test_holdout.csv
    challenge_validation_demo.csv
```

The `.gitignore` keeps `data/raw` and `data/processed` out of git. Manifest CSVs are safe to version because they document the exact split and label mapping without storing image files.

## Download Options

### Option A: CIFAKE From Kaggle

Use Kaggle for the first smoke-test dataset.

Prerequisites:

- Kaggle account
- Kaggle API token at `~/.kaggle/kaggle.json`

Planned command:

```bash
kaggle datasets download \
  -d birdy654/cifake-real-and-ai-generated-synthetic-images \
  -p data/raw/cifake \
  --unzip
```

Expected folder idea:

```text
data/raw/cifake/
  train/
    REAL/
    FAKE/
  test/
    REAL/
    FAKE/
```

If the exact folder names differ after download, update the manifest script instead of renaming labels by hand.

### Option B: SID_Set From Hugging Face

Use Hugging Face Datasets for SID_Set.

Prerequisites:

```bash
pip install datasets huggingface_hub
```

Planned inspection command:

```bash
python - <<'PY'
from datasets import load_dataset

ds = load_dataset("saberzl/SID_Set", split="train", streaming=True)
row = next(iter(ds))
print(row.keys())
print({k: row[k] for k in row.keys() if k != "image" and k != "mask"})
PY
```

Important:

- Confirm the available splits before downloading a full local copy.
- Confirm label meanings before manifesting.
- Use label `0` and label `1` first.
- Exclude label `2` from the first binary baseline.

### Option C: WildFake From ModelScope

Use ModelScope or the download link referenced by the WildFake repository.

Prerequisites:

```bash
pip install modelscope
```

Planned command pattern:

```bash
modelscope download \
  --dataset hy2628982280/WildFake \
  --local_dir data/raw/wildfake
```

After download, inspect the folder structure before creating a manifest:

```bash
find data/raw/wildfake -maxdepth 3 -type d | sort | head -80
find data/raw/wildfake -maxdepth 3 -type f | sort | head -80
```

Expected label idea:

```text
Real -> 0
GAN_based -> 1
Diffusion_based -> 1
Other_based -> 1
```

Do not assume this blindly. Verify actual folder names and split annotations after download.

## Challenge Validation-Demo Subset

The problem statement says the validation-demo subset is for demonstration only and must not be used during training.

Treat it as locked:

```text
data/raw/challenge_validation_demo/
```

Allowed use:

- Final demo-style evaluation
- Reporting reference benchmark behavior

Not allowed:

- Training
- Hyperparameter tuning
- Threshold selection
- Calibration
- Choosing between model architectures

## Manifest Creation Plan

Every image we use should be represented by one CSV row:

```text
image_path,label,source_dataset,generator,split,license_notes
```

Example rows:

```csv
image_path,label,source_dataset,generator,split,license_notes
data/raw/cifake/train/REAL/0001.jpg,0,cifake,cifar10,train,MIT-like CIFAR-10/CIFAKE terms
data/raw/cifake/train/FAKE/0001.jpg,1,cifake,stable_diffusion_1_4,train,MIT-like CIFAR-10/CIFAKE terms
data/raw/sid_set/train/full_synthetic/example.png,1,sid_set,unknown_full_synthetic,train,CC-BY-4.0
```

Split discipline:

- `train.csv`: training only
- `val.csv`: validation, calibration, threshold selection
- `test_clean.csv`: clean held-out test set
- `test_holdout.csv`: unseen source or generator holdout
- `challenge_validation_demo.csv`: locked demo benchmark only

## Immediate Next Step

Do this before writing training code:

1. Confirm CIFAKE exists locally under `data/raw/cifake/archive`.
2. Create labelled manifests:

```bash
python scripts/make_cifake_manifest.py
```

3. Inspect the generated files:

```bash
head data/manifests/train.csv
head data/manifests/val.csv
head data/manifests/test_clean.csv
head data/manifests/cifake_sample.csv
```

4. Use the sample manifest to test image loading and transform code.
5. Then add SID_Set and WildFake manifests.

This keeps us moving without accidentally contaminating the real evaluation setup.
