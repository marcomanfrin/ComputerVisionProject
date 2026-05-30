# Plant Disease Detection — Results Summary

_Generated: 2026-05-30 20:17_

Dataset: **Plant Village** (~54.000 immagini, 38 classi)  
Split: 70% train / 15% val / 15% test (stratified, seed=42)

## Tabella Metriche (test set)

| Versione | Approccio | Accuracy | Precision | Recall | F1 |
|----------|-----------|:--------:|:---------:|:------:|:--:|
| V1 — HOG + SVM | Shallow Learning | 74.39% | 74.62% | 74.39% | 74.26% |
| V2 — Custom CNN | Deep Learning from scratch | 99.66% | 99.66% | 99.66% | 99.66% |
| V3 — ResNet50 Transfer Learn. | Supervised TL + fine-tuning | 99.87% | 99.88% | 99.87% | 99.87% |
| V4 — DINOv3 + Linear Probe | SSL Foundation Model frozen | 98.35% | 98.39% | 98.35% | 98.35% |

## Key Findings

- **Migliore overall:** V3 — ResNet50 Transfer Learn. — accuracy 99.87%, F1 99.87%
- **Gap V1 → V3:** +25.5 punti di accuracy. La feature extraction manuale (HOG) collassa rispetto a feature apprese end-to-end.
- **V4 SSL competitivo senza training:** 98.35% di accuracy con backbone **completamente congelato** (0 parametri trainable nel backbone). Solo k-NN o un logistic regression sopra le feature.
- **Transfer learning più veloce di un custom CNN:** V3 converge in 19 epoch (~128 min) vs V2 80 epoch (~294 min).
- **V4 k-NN vs Linear Probe:** k-NN 98.25% vs LinProbe 98.35%. Il linear probe sfrutta meglio la struttura globale dello spazio di feature.

## Raccomandazioni per il Deployment

| Scenario | Versione consigliata | Motivazione |
|----------|----------------------|-------------|
| Max accuracy, dataset fisso | **V3 (ResNet50 TL)** | Best score assoluto, parametri ragionevoli (~24M totali). |
| Onboarding rapido di nuove classi | **V4 (DINOv3 + linear probe)** | Backbone congelato, basta riaddestrare il logistic regression in secondi. |
| Edge / CPU-only / interpretabilità | **V1 (HOG+SVM)** | Modello piccolo, ispezionabile, ma accuracy ~75%. |
| Custom architecture per ricerca / didattica | **V2 (CNN from scratch)** | Pieno controllo dell'architettura, ottimo per oral exam. |

## Riproducibilità

- Seed fissato (`random_state=42`) su split, sklearn, torch.
- Notebook ordinati: `00_setup_and_data` → `06_prepare_documentation`.
- Dipendenze in `requirements.txt`. V4 richiede `transformers` + login HuggingFace per DINOv3.
- Checkpoints in `results/models/<version>/` (esclusi da git tramite `.gitignore`).
- Cache embedding V4 in `results/models/v4_dinov3_probe/embeddings/*.npz` (rigenerabili in ~13 min).
