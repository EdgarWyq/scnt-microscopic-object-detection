# Portfolio Summary

## Project

**SCNT Microscopic Object Detection**  
YOLO-based detection system for microscopic manipulation images, detecting injection needles, holding needles, and oocytes.

## What I Built

I built an end-to-end object detection workflow instead of only training a single model:

- YOLO dataset validation for image-label matching and normalized bbox checking.
- Source-domain baseline training and target-domain validation.
- Microscopic image augmentation for domain generalization.
- Pseudo-label generation and self-training experiments.
- Analysis of confirmation bias in pseudo labels.
- Fast manual image review tool for active sample selection.
- Lightweight OpenCV YOLO annotation tool.
- Few-shot target-domain retraining with 50 manually refined samples.
- Final evaluation on 479 held-out target-domain images.
- Batch prediction visualization for qualitative review.

## Technical Stack

- Python
- Ultralytics YOLO / YOLO11s
- PyTorch
- OpenCV
- pandas / matplotlib / tqdm
- Windows + PyCharm + RTX 4060 Laptop GPU

## Core Challenge

The source and target microscopic domains differ in scale, color, contrast, background texture, and object morphology. The hardest issue was the confusion between `holding_needle` and `injection_needle`, especially when target-domain needles appeared in shapes or orientations underrepresented in the source domain.

## Method Evolution

1. **Source-only YOLOv8n baseline**  
   Established the lower-bound performance and exposed the domain gap.

2. **YOLO11s + source-domain augmentation**  
   Improved robustness to color, scale, and small-object shifts.

3. **Pseudo-label self-training**  
   Implemented an unsupervised adaptation pipeline, but found that systematic mistakes could be reinforced by pseudo labels.

4. **Manual high-value sample loop**  
   Selected and refined 50 target-domain samples based on model errors, then retrained YOLO11s with sampled source data and target-domain annotations.

## Results

| Method | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source plain YOLOv8n | 0.2474 | 0.2258 | 0.0788 | 0.1840 | 0.0901 |
| YOLO11s + source augmentation | 0.6938 | 0.6574 | 0.9608 | 0.7707 | 0.4842 |
| Manual-50 YOLO11s retrain | 0.9943 | 0.9774 | 0.9914 | 0.9877 | 0.7183 |

The final result is a few-shot supervised target-domain adaptation experiment, evaluated on 479 held-out target-domain images.

## Useful Ablation

I also tested morphology-based post-processing to reduce holding/injection confusion. On full SCNT-Target using a custom evaluator at `conf=0.25`:

| Mode | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.3904 | 0.0784 | 0.9145 | 0.4611 | 0.3119 |
| postprocess | 0.4592 | 0.4987 | 0.9122 | 0.6234 | 0.4351 |

This showed that application-layer rules can partially fix systematic class confusion, but better annotated target-domain data is more robust.

## Representative Visualizations

See [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md) for raw predictions, post-processed predictions, and final model examples.

## Why This Project Is Relevant for Internships

This project demonstrates practical machine learning engineering skills:

- Building reliable data pipelines.
- Designing fair train/eval splits.
- Debugging model failures with visual evidence.
- Avoiding misleading metrics and clearly separating unsupervised and supervised settings.
- Turning error analysis into targeted data collection.
- Writing reusable training, validation, annotation, and visualization scripts.
- Presenting results honestly with limitations and ablations.
