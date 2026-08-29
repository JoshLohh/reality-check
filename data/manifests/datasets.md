# Dataset Inventory

## CIFAKE

- Source: Kaggle dataset `birdy654/cifake-real-and-ai-generated-synthetic-images`
- Local root: `data/raw/cifake/archive`
- Label mapping: `REAL -> 0`, `FAKE -> 1`
- Real generator/source: `cifar10`
- AIGC generator/source: `stable_diffusion`
- Role: smoke test and initial baseline data
- Caveat: images are 32x32, so CIFAKE alone is not enough for the final robustness claim
- Hackathon note: no challenge validation-demo data is included here
