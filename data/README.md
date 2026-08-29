# Data Directory

Keep downloaded datasets and generated artifacts out of git.

Expected local layout:

```text
data/
  raw/
    cifake/
    sid_set/
    wildfake/
    challenge_validation_demo/
  processed/
    sample_images/
  manifests/
    train.csv
    val.csv
    test_clean.csv
    test_holdout.csv
    challenge_validation_demo.csv
```

Rules:

- Do not commit raw datasets.
- Do not train on `challenge_validation_demo`.
- Do not tune thresholds or calibration on test data.
- Record each dataset's source and license before using it.
