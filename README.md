# SCNT Microscopic Object Detection

YOLO-based object detection for microscopic SCNT manipulation images, with source-domain training, domain generalization, pseudo-label experiments, and a few-shot target-domain annotation loop.

This repository is prepared as a portfolio project: it focuses on the engineering workflow, experiment design, error analysis, and reproducible scripts. The dataset, trained weights, and full prediction outputs are intentionally not committed because they may be large or restricted.

## Task

Detect three object categories in microscopic manipulation images:

| ID | Class | Description |
| --- | --- | --- |
| 0 | `injection_needle` | injection needle |
| 1 | `holding_needle` | holding needle |
| 2 | `oocyte` | oocyte |

Original data layout:

```text
dataset/
+-- SCNT/
    +-- SCNT-Source/
    |   +-- images/
    |   +-- labels/
    +-- SCNT-Target/
        +-- images/
        +-- labels/
```

Experimental data usage:

| Split | Images | Usage |
| --- | ---: | --- |
| SCNT-Source | 2785 | labeled source-domain training data |
| SCNT-Target | 529 | target-domain images |
| Manual target train | 50 | manually refined target-domain samples |
| Final eval | 479 | held-out target-domain evaluation set |
| ManualMix train | 1871 | sampled source + source augmentations + weighted manual target samples |

Important: target-domain labels are only used for final evaluation and error analysis. The final few-shot experiment uses 50 manually refined target-domain samples and evaluates on the remaining 479 images.

## Why This Project Is Interesting

The task is harder than a standard YOLO demo:

- Domain gap between source and target microscopic images.
- Small and elongated objects, especially needles.
- Similar visual patterns between `holding_needle` and `injection_needle`.
- Color and illumination shift, including warm/orange backgrounds.
- Oocyte false positives caused by bubbles, debris, or dark particles.
- Need for a practical data loop rather than blindly increasing the training set.

The final solution was not just "train YOLO once". It became a complete detection workflow:

1. Dataset validation.
2. Source-only baseline.
3. Microscopic image augmentation.
4. Pseudo-label self-training and confirmation-bias analysis.
5. Manual error-driven image selection.
6. Lightweight YOLO annotation tooling.
7. Few-shot target-domain retraining.
8. Held-out final evaluation and visualization.

## Representative Results

### Final Manual-50 Model

| Example 1 | Example 2 | Example 3 |
| --- | --- | --- |
| ![](docs/assets/representative_results/manual50_final/1293.jpg) | ![](docs/assets/representative_results/manual50_final/1470.jpg) | ![](docs/assets/representative_results/manual50_final/1501.jpg) |

### Source Augmentation Raw vs Post-Processing

| Raw prediction | Application-layer post-processing |
| --- | --- |
| ![](docs/assets/representative_results/source_aug_raw/1293.jpg) | ![](docs/assets/representative_results/source_aug_postprocess/1293.jpg) |
| ![](docs/assets/representative_results/source_aug_raw/1649.jpg) | ![](docs/assets/representative_results/source_aug_postprocess/1649.jpg) |

More examples and ablations are available in [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md).

## Main Results

| Experiment | Eval split | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Source plain YOLOv8n | target_eval | 0.2474 | 0.2258 | 0.0788 | 0.1840 | 0.0901 |
| YOLO11s + source augmentation | target_eval | 0.6938 | 0.6574 | 0.9608 | 0.7707 | 0.4842 |
| YOLO11s raw reference | full SCNT-Target | 0.6616 | 0.6448 | 0.9499 | 0.7521 | 0.4864 |
| Manual-50 YOLO11s retrain | final_eval 479 | 0.9943 | 0.9774 | 0.9914 | 0.9877 | 0.7183 |

The final row is a few-shot supervised target-domain adaptation experiment, not a strictly unsupervised domain adaptation result.

## Valuable Ablation: Source Augmentation + Post-Processing

Before using manual target-domain labels, I also tested an application-layer morphology post-processing strategy to reduce `holding_needle` / `injection_needle` confusion.

On full SCNT-Target with a custom evaluator at `conf=0.25`:

| Mode | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.3904 | 0.0784 | 0.9145 | 0.4611 | 0.3119 |
| postprocess | 0.4592 | 0.4987 | 0.9122 | 0.6234 | 0.4351 |

This showed that morphology rules can partially correct systematic needle-class confusion, but they are not a substitute for better target-domain training data. The final model therefore uses 50 high-value manually refined samples.

## Repository Structure

```text
configs/
  scnt_source.yaml
  scnt_source_aug.yaml
  scnt_pseudo.yaml
  scnt_manual_mix.yaml
scripts/
  check_dataset.py
  split_target.py
  train_baseline.py
  val_model.py
  predict_visualize.py
  generate_pseudo_labels.py
  build_pseudo_dataset.py
  train_pseudo_adapt.py
  build_augmented_source.py
  review_select_images.py
  annotate_yolo.py
  build_manual_finetune_dataset.py
docs/
  EXPERIMENT_RESULTS.md
  assets/representative_results/
REPORT.md
PROJECT_SUMMARY.md
requirements.txt
```

## Key Scripts

| Script | Purpose |
| --- | --- |
| `scripts/check_dataset.py` | validate image-label matching and YOLO label format |
| `scripts/train_baseline.py` | unified YOLO training entry point |
| `scripts/val_model.py` | validation and metric logging |
| `scripts/build_augmented_source.py` | offline microscopic image augmentation |
| `scripts/generate_pseudo_labels.py` | high-confidence pseudo-label generation |
| `scripts/eval_postprocess.py` | raw vs morphology post-processing evaluation |
| `scripts/review_select_images.py` | fast manual image selection for active learning |
| `scripts/annotate_yolo.py` | lightweight OpenCV YOLO annotator |
| `scripts/build_manual_finetune_dataset.py` | build final few-shot target adaptation dataset |

## Setup

```powershell
pip install -r requirements.txt
```

Main experiment environment:

```text
Python 3.12
Ultralytics 8.4.51
PyTorch 2.10.0+cu126
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## Reproduce the Workflow

### 1. Check dataset

```powershell
python scripts/check_dataset.py
```

Output:

```text
outputs/dataset_check.txt
```

### 2. Train source baseline

```powershell
python scripts/train_baseline.py --data configs/scnt_source.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --project runs/scnt --name baseline
```

### 3. Train YOLO11s source-augmentation model

```powershell
python scripts/train_baseline.py --data configs/scnt_source_aug.yaml --model yolo11s.pt --epochs 50 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name source_aug_yolo11s_compact_smallobj_960 --exist-ok --quiet
```

### 4. Select target-domain samples for manual refinement

```powershell
python scripts/review_select_images.py --target-count 50 --reset-selection --overwrite-export
```

### 5. Pre-label selected images

```powershell
python scripts/prelabel_fewshot.py --model runs/scnt/source_aug_yolo11s_compact_smallobj_960/weights/best.pt --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels --imgsz 960 --device 0 --predict-conf 0.25 --conf-class-0 0.25 --conf-class-1 0.25 --conf-class-2 0.25 --overwrite-labels
```

### 6. Manually refine labels

```powershell
python scripts/annotate_yolo.py --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels
```

### 7. Build final mixed training dataset

```powershell
python scripts/build_manual_finetune_dataset.py --source-frac 0.20 --aug-per-source 2 --manual-repeat 4 --seed 42 --overwrite
```

### 8. Train final model

```powershell
python scripts/train_baseline.py --data configs/scnt_manual_mix.yaml --model yolo11s.pt --epochs 80 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name manual50_mix_yolo11s_from_pretrain_960 --patience 20 --hsv-h 0.005 --hsv-s 0.2 --hsv-v 0.15 --scale 0.15 --translate 0.05 --mosaic 0.2 --mixup 0.0 --fliplr 0.0 --flipud 0.0 --close-mosaic 15 --exist-ok --quiet
```

In the recorded run, the best checkpoint appeared around epoch 15 and training was stopped early.

### 9. Evaluate final model

```powershell
python scripts/val_model.py --model runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt --data configs/scnt_manual_mix.yaml --name manual50_mix_yolo11s_from_pretrain_960_earlystop_best --imgsz 960 --batch 2 --rect --device 0 --summary-csv outputs/experiments_summary.csv --exist-ok --quiet
```

### 10. Visualize final predictions

```powershell
python scripts/predict_visualize.py --model runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt --source dataset/SCNT-ManualSelf/final_eval/images --output outputs/visualizations/manual50_mix_yolo11s_from_pretrain_960_final_eval_raw --max-images 0 --conf 0.25 --imgsz 960 --batch 1 --rect --device 0 --exist-ok --quiet
```

## Notes for Reviewers

- This repository does not include the dataset or trained weights.
- Representative visualizations are included only for portfolio demonstration.
- The final result uses 50 manually refined target-domain samples, so it should be described as few-shot supervised target-domain adaptation.
- The pseudo-label experiments are kept because they show a realistic failure mode: confirmation bias can amplify systematic model errors.

## Documents

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md): concise portfolio summary.
- [REPORT.md](REPORT.md): course-style report in Chinese.
- [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md): ablations and representative visualizations.
