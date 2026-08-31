# Error Analysis Note — Reality Check

## Model and evaluation setup

The selected checkpoint is a pretrained EfficientNet-B0 with a frozen backbone and a binary classifier head. It was trained for 15 epochs on an 80% SID_Set / 20% CIFAKE mixture; epoch 13 achieved the best validation ROC AUC (0.9640). No WildFake images were used in training.

The model produces a sigmoid **AI score** between 0 and 1. These scores are not explicitly calibrated probabilities. Threshold-dependent error counts use a fixed threshold of 0.5, while ROC AUC evaluates ranking across every threshold. Robust AUC is the mean AUC across 14 deterministic conditions covering JPEG compression, blur, resizing, noise, colour jitter and centre cropping.

## Results

| Evaluation set | Clean AUC | Mean robust AUC | Worst condition | Clean accuracy | Clean false positives | Clean false negatives |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| SID_Set | 0.9750 | 0.9706 | noise σ=0.10: 0.9571 | 0.9210 | 49 | 30 |
| CIFAKE | 0.8750 | 0.8080 | blur σ=2.0: 0.6666 | 0.8050 | 82 | 113 |
| WildFake slice (deduplicated) | 0.4766 | 0.4145 | noise σ=0.05: 0.2908 | 0.5256 | 149 | 324 |

The strong SID_Set result shows that the detector is robust within its dominant training domain. Performance falls on CIFAKE and collapses on the evaluated WildFake slice, demonstrating that transformation robustness and cross-dataset generalisation are different problems.

## Representative false positives

Authentic images were incorrectly assigned high AI scores when they contained characteristics commonly associated with generated content:

- A polished jet-ski product photograph (AI score 0.771) used glossy surfaces, controlled lighting and a render-like dark background.
- An HDR-like kitchen photograph (0.793) contained strong processing and motion ghosting.
- A food photograph with text overlay and stylised framing (0.805) resembled edited social-media content.
- A distant dolphin photograph (0.766) contained haze, repetitive water texture and a very small subject.

These examples suggest that the detector sometimes treats photographic style, editing and low-texture regions as evidence of generation. This is a practical false-positive risk for product photography, professional portraits and edited authentic content.

## Representative false negatives

AI-generated images were incorrectly assigned low AI scores when they closely reproduced familiar photographic cues:

- A street scene containing a parking meter and car received 0.201. Reflections and depth of field were convincing, although small text was malformed.
- A vehicle product image received 0.228 because its shadows, reflections and branding appeared coherent.
- A STOP-sign image received 0.252; familiar geometry, readable large text and realistic background blur made it appear authentic.
- A decorative sculpture received 0.252; realistic material detail and studio lighting outweighed its unusual semantics.

Several generated images contained semantic inconsistencies or malformed small text, but the frozen image classifier did not reliably exploit them. The model appears more sensitive to texture and photographic style than to scene meaning.

## Transformation-induced failures

The class direction of failures matters:

- On SID_Set, noise σ=0.10 created 90 new errors among images that were correct when clean; 86 were false negatives.
- On CIFAKE, blur σ=2.0 created 296 new errors, including 205 false negatives and 91 false positives.
- CIFAKE resizing to 0.25× created 279 new errors.
- CIFAKE noise σ=0.05 created 285 new errors, of which 284 were false negatives.

Noise therefore tends to suppress the AI score and makes synthetic images look authentic to the detector. Strong blur and downscaling damage both classes and reduce their separability.

## Trade-offs and limitations

- Robust augmentation preserves strong SID_Set AUC, but it does not guarantee transfer to unseen sources and generators.
- Lowering the 0.5 threshold could recover more AI images under noise, but would increase false accusations against authentic images.
- A different threshold cannot repair the below-random WildFake AUC because AUC measures ranking, not threshold choice.
- The WildFake evaluation contains 1,000 rows but only 997 unique paths. It also contains only AFHQ authentic images and DDIM fakes, so it measures one domain pair rather than broad WildFake generalisation.
- Qualitative inspection was performed on 16 high-confidence SID_Set errors. CIFAKE and WildFake conclusions are quantitative because their raw evaluated images were not included in the supplied result archive.
- Visual explanations are hypotheses. Grad-CAM or controlled image edits would be needed to establish which regions causally drive each decision.

## Improvements

The next model should include authentic hard negatives such as retouched portraits, HDR images, product photography, screenshots and text overlays. Training should also include photorealistic hard positives from multiple generator families. Evaluation manifests must enforce unique paths and stratify across real sources and fake generators. A frequency branch, OCR/semantic consistency features, or partial backbone fine-tuning could complement the current texture-focused classifier.
