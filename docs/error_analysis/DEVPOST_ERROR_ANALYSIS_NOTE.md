## Error analysis

We evaluated the epoch-13 SID80/CIFAKE20 EfficientNet-B0 checkpoint on clean images and 14 redistribution conditions. The model achieved 0.975 clean AUC and 0.971 mean robust AUC on SID_Set, but performance fell to 0.875/0.808 on CIFAKE. This shows that robustness within the dominant training domain does not guarantee cross-dataset generalisation.

Visual inspection of high-confidence SID_Set errors revealed two recurring patterns. Authentic false positives were frequently highly processed or visually stylised: product photography, HDR-like interiors, text-overlaid food images and low-detail wildlife scenes. AI false negatives were polished generations with plausible camera lighting, coherent depth of field and realistic material texture. Some contained malformed small text or semantic inconsistencies, suggesting that the frozen EfficientNet detector relies more on texture and photographic style than semantic reasoning.

Transformation failures were class-dependent. SID noise at σ=0.10 created 90 new errors from previously correct images, including 86 false negatives. On CIFAKE, blur σ=2.0 created 296 new errors, 0.25× resizing created 279, and noise σ=0.05 created 285; 284 of the noise failures were AI images predicted as authentic. Lowering the threshold could improve recall under noise, but would increase false positives on authentic images.

The evaluated WildFake slice performed poorly, but it contained only AFHQ real images versus DDIM fakes and included three duplicate paths. We therefore treat it as a domain-pair diagnostic rather than a broad WildFake benchmark. Future work will use deduplicated, generator-stratified evaluation and more diverse hard-negative training.
