# Plant Disease Detection — Results Summary

_Generated: 2026-05-31 11:44_

Dataset: **New Plant Diseases Dataset** (augmented PlantVillage, 87,867 images, 38 classes)  
Split: train 70,295 / val 8,777 / test 8,795 (≈ 80/10/10, valid/ split 50/50 with seed=42)

## Metrics Table (test set)

| Version | Approach | Accuracy | Precision | Recall | F1 |
|----------|-----------|:--------:|:---------:|:------:|:--:|
| V1 — HOG + SVM | Shallow Learning | 74.39% | 74.62% | 74.39% | 74.26% |
| V2 — Custom CNN | Deep Learning from scratch | 99.66% | 99.66% | 99.66% | 99.66% |
| V3 — ResNet50 Transfer Learn. | Supervised TL + fine-tuning | 99.87% | 99.88% | 99.87% | 99.87% |
| V4 — DINOv3 + Linear Probe | SSL Foundation Model frozen | 98.35% | 98.39% | 98.35% | 98.35% |

## Key Findings

- **Best overall:** V3 — ResNet50 Transfer Learn. — accuracy 99.87%, F1 99.87%
- **Gap V1 → V3:** +25.5 accuracy points. Manual feature extraction (HOG) collapses compared to features learned end-to-end.
- **V4 SSL competitive without training:** 98.35% accuracy with a **completely frozen** backbone (0 trainable parameters in the backbone). Just k-NN or a logistic regression on top of the features.
- **Transfer learning faster than a custom CNN:** V3 converges in 19 epochs (~128 min) vs V2 80 epochs (~294 min).
- **V4 k-NN vs Linear Probe:** k-NN 98.25% vs LinProbe 98.35%. The linear probe exploits the global structure of the feature space better.

> 🔬 **Leak analysis (notebook 07):** the dataset is augmented offline, so augmented copies of the same source leaf can land in both train and val/test. We quantified the leak by stripping augmentation suffixes (`_90deg`, `_flipLR`, …) to recover source identities. Result: **63.3%** of test source IDs are also in train (5,732 / 8,795 images). However, re-evaluating the trained checkpoints on the leak-free 3,063-image subset shows the clean accuracy is within ±0.16 pts of the published full-test number for every model — V3 and V4 are in fact marginally _higher_ on the clean subset. **The leak is statistically irrelevant** for these results; the models learned a genuine representation, not augmented duplicates. Details in `Technical_Analysis.md` §4.5.

## Deployment Recommendations

| Scenario | Recommended version | Rationale |
|----------|----------------------|-------------|
| Max accuracy, fixed dataset | **V3 (ResNet50 TL)** | Best absolute score, reasonable parameters (~24M total). |
| Fast onboarding of new classes | **V4 (DINOv3 + linear probe)** | Frozen backbone, just retrain the logistic regression in seconds. |
| Edge / CPU-only / interpretability | **V1 (HOG+SVM)** | Small, inspectable model, but ~75% accuracy. |
| Custom architecture for research / teaching | **V2 (CNN from scratch)** | Full control of the architecture, great for the oral exam. |

## Reproducibility

- Fixed seed (`random_state=42`) on split, sklearn, torch.
- Ordered notebooks: `00_setup_and_data` → `06_prepare_documentation`.
- Dependencies in `requirements.txt`. V4 requires `transformers` + HuggingFace login for DINOv3.
- Checkpoints in `results/models/<version>/` (excluded from git via `.gitignore`).
- V4 embedding cache in `results/models/v4_dinov3_probe/embeddings/*.npz` (regenerable in ~13 min).
