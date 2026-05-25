# SCNT 显微目标检测

这是一个面向显微操作图像的目标检测项目，检测对象包括注射针、吸持针和卵细胞。项目不是简单调用 YOLO 训练一次，而是完整覆盖了数据检查、baseline、域泛化增强、伪标签实验、误差分析、人工高价值样本筛选、轻量标注工具、少样本目标域重训和最终验证。

本仓库主要用于实习求职展示，重点展示机器学习工程能力、实验设计能力和错误分析能力。数据集、训练权重和完整预测输出未上传，避免仓库过大以及潜在数据授权问题。

## 项目亮点

- 使用 Ultralytics YOLO / YOLO11s 完成显微图像三类目标检测。
- 针对源域和目标域差异，设计显微图像增强和小目标相关增强。
- 实现伪标签自训练流程，并分析 confirmation bias 对模型的负面影响。
- 针对 holding needle 与 injection needle 混淆问题，做了可视化误差分析和后处理消融。
- 自己实现快速筛图脚本和 OpenCV YOLO 标注工具，提高人工精标效率。
- 只人工精修 50 张目标域图像，在 479 张独立验证图上完成最终评估。
- 项目代码包含训练、验证、可视化、伪标签、数据构建和实验对比脚本。

## 检测类别

| ID | 类别 | 中文说明 |
| --- | --- | --- |
| 0 | `injection_needle` | 注射针 |
| 1 | `holding_needle` | 吸持针 |
| 2 | `oocyte` | 卵细胞 |

## 数据与实验划分

原始数据结构：

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

本项目中的主要数据规模：

| 数据部分 | 图像数 | 用途 |
| --- | ---: | --- |
| SCNT-Source | 2785 | 源域有标签训练数据 |
| SCNT-Target | 529 | 目标域图像 |
| Manual target train | 50 | 人工精修的目标域训练样本 |
| Final eval | 479 | 目标域独立最终验证集 |
| ManualMix train | 1871 | 源域采样 + 源域增强 + 目标域精标样本重复采样 |

说明：目标域真实标签只用于最终验证和误差分析。最终方案使用了 50 张人工精修目标域图像，因此属于“小样本目标域监督适应”，不是严格无监督域自适应。

## 代表性效果

### 最终 Manual-50 模型

| 示例 1 | 示例 2 | 示例 3 |
| --- | --- | --- |
| ![](docs/assets/representative_results/manual50_final/1293.jpg) | ![](docs/assets/representative_results/manual50_final/1470.jpg) | ![](docs/assets/representative_results/manual50_final/1501.jpg) |

### 源域增强模型：原始预测 vs 应用层后处理

| 原始预测 | 后处理预测 |
| --- | --- |
| ![](docs/assets/representative_results/source_aug_raw/1293.jpg) | ![](docs/assets/representative_results/source_aug_postprocess/1293.jpg) |
| ![](docs/assets/representative_results/source_aug_raw/1649.jpg) | ![](docs/assets/representative_results/source_aug_postprocess/1649.jpg) |

更多可视化和消融结果见：[docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)

## 主要结果

| 实验 | 验证集 | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Source plain YOLOv8n | target_eval | 0.2474 | 0.2258 | 0.0788 | 0.1840 | 0.0901 |
| YOLO11s + source augmentation | target_eval | 0.6938 | 0.6574 | 0.9608 | 0.7707 | 0.4842 |
| YOLO11s raw reference | full SCNT-Target | 0.6616 | 0.6448 | 0.9499 | 0.7521 | 0.4864 |
| Manual-50 YOLO11s retrain | final_eval 479 | 0.9943 | 0.9774 | 0.9914 | 0.9877 | 0.7183 |

最终模型使用 50 张目标域人工精标样本，并在剩余 479 张目标域图像上验证。

## 有价值的探索：增强模型 + 后处理

在引入人工目标域标注前，我还测试了“源域增强模型 + 应用层形态后处理”。该方法不重新训练模型，而是在推理阶段根据框的形态特征，把部分疑似 holding needle 的 injection 预测重标为 holding，并过滤异常大的 oocyte 框。

在 full SCNT-Target 上使用自定义 evaluator，`conf=0.25`：

| Mode | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.3904 | 0.0784 | 0.9145 | 0.4611 | 0.3119 |
| postprocess | 0.4592 | 0.4987 | 0.9122 | 0.6234 | 0.4351 |

这个实验说明，holding/injection 的系统性混淆可以被形态规则部分纠正，但它不如直接补充高质量目标域标注稳定，因此最终主方案采用 50 张精标样本重训。

## 方法流程

1. **数据检查**
   检查图像和标签是否一一对应，YOLO 标签格式是否合法，类别和坐标是否正确。

2. **源域 baseline**
   使用 `SCNT-Source` 有标签数据训练 YOLOv8n baseline，验证 domain gap。

3. **域泛化增强**
   使用 YOLO11s 和显微图像相关增强，缓解颜色、尺度、小目标和背景差异。

4. **伪标签自训练**
   使用目标域无标签图像生成高置信度伪标签，但发现系统性错误会被伪标签放大。

5. **人工高价值样本筛选**
   根据模型误检漏检结果，筛选 holding/injection 混淆、小目标、暖色背景、模糊和截断目标等高价值图像。

6. **少样本目标域重训**
   人工精修 50 张目标域样本，与源域采样和源域增强数据混合，从 YOLO11s 重新训练。

7. **最终验证与可视化**
   在 479 张未参与训练的目标域图像上验证，并输出全部预测可视化。

## 仓库结构

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

## 核心脚本

| 脚本 | 功能 |
| --- | --- |
| `scripts/check_dataset.py` | 检查数据集格式和标签合法性 |
| `scripts/train_baseline.py` | 统一 YOLO 训练入口 |
| `scripts/val_model.py` | 验证模型并记录指标 |
| `scripts/build_augmented_source.py` | 构建显微图像离线增强数据 |
| `scripts/generate_pseudo_labels.py` | 生成目标域伪标签 |
| `scripts/eval_postprocess.py` | 对比 raw prediction 和后处理结果 |
| `scripts/review_select_images.py` | 快速人工筛选高价值目标域样本 |
| `scripts/annotate_yolo.py` | OpenCV 轻量 YOLO 标注工具 |
| `scripts/build_manual_finetune_dataset.py` | 构建最终小样本目标域训练集 |

## 环境安装

```powershell
pip install -r requirements.txt
```

主要实验环境：

```text
Python 3.12
Ultralytics 8.4.51
PyTorch 2.10.0+cu126
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## 运行流程

### 1. 数据检查

```powershell
python scripts/check_dataset.py
```

输出：

```text
outputs/dataset_check.txt
```

### 2. 训练源域 baseline

```powershell
python scripts/train_baseline.py --data configs/scnt_source.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --project runs/scnt --name baseline
```

### 3. 训练 YOLO11s 源域增强模型

```powershell
python scripts/train_baseline.py --data configs/scnt_source_aug.yaml --model yolo11s.pt --epochs 50 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name source_aug_yolo11s_compact_smallobj_960 --exist-ok --quiet
```

### 4. 快速筛选目标域人工精标样本

```powershell
python scripts/review_select_images.py --target-count 50 --reset-selection --overwrite-export
```

### 5. 对选中图片生成预标注

```powershell
python scripts/prelabel_fewshot.py --model runs/scnt/source_aug_yolo11s_compact_smallobj_960/weights/best.pt --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels --imgsz 960 --device 0 --predict-conf 0.25 --conf-class-0 0.25 --conf-class-1 0.25 --conf-class-2 0.25 --overwrite-labels
```

### 6. 人工精修标签

```powershell
python scripts/annotate_yolo.py --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels
```

### 7. 构建最终训练集

```powershell
python scripts/build_manual_finetune_dataset.py --source-frac 0.20 --aug-per-source 2 --manual-repeat 4 --seed 42 --overwrite
```

### 8. 训练最终模型

```powershell
python scripts/train_baseline.py --data configs/scnt_manual_mix.yaml --model yolo11s.pt --epochs 80 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name manual50_mix_yolo11s_from_pretrain_960 --patience 20 --hsv-h 0.005 --hsv-s 0.2 --hsv-v 0.15 --scale 0.15 --translate 0.05 --mosaic 0.2 --mixup 0.0 --fliplr 0.0 --flipud 0.0 --close-mosaic 15 --exist-ok --quiet
```

实际训练中，最佳 checkpoint 出现在约第 15 轮，后续指标下降，因此提前停止。

### 9. 验证最终模型

```powershell
python scripts/val_model.py --model runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt --data configs/scnt_manual_mix.yaml --name manual50_mix_yolo11s_from_pretrain_960_earlystop_best --imgsz 960 --batch 2 --rect --device 0 --summary-csv outputs/experiments_summary.csv --exist-ok --quiet
```

### 10. 可视化最终预测

```powershell
python scripts/predict_visualize.py --model runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt --source dataset/SCNT-ManualSelf/final_eval/images --output outputs/visualizations/manual50_mix_yolo11s_from_pretrain_960_final_eval_raw --max-images 0 --conf 0.25 --imgsz 960 --batch 1 --rect --device 0 --exist-ok --quiet
```

## 面试时可以如何介绍

这个项目可以概括为：

> 我做了一个显微操作场景下的三类目标检测项目。难点是源域和目标域差异明显，holding needle 和 injection needle 容易混淆，小目标和模糊目标漏检严重。我先建立 YOLO baseline，再做显微图像增强和伪标签实验，发现伪标签会放大系统性错误。最后我实现了快速筛图和标注工具，只精修 50 张高价值目标域样本，与源域采样增强数据混合训练，在 479 张独立目标域测试图上达到 mAP50 0.9877。

这个项目体现的能力：

- 目标检测工程落地。
- 数据清洗和格式检查。
- 模型训练、验证和可视化闭环。
- 错误分析和主动选样。
- 少样本目标域适应。
- 对实验设置和指标可信度有清晰边界。

## 注意事项

- 本仓库不包含数据集、训练权重和完整输出结果。
- 代表性图片仅用于项目展示。
- 最终结果使用了 50 张目标域人工精标样本，应表述为小样本目标域监督适应。
- 伪标签实验保留为探索过程，用于说明 confirmation bias 的真实问题。

## English Summary

This project builds a YOLO-based microscopic object detection workflow for SCNT manipulation images. It detects injection needles, holding needles, and oocytes. The final pipeline combines source-domain training, domain-specific augmentation, pseudo-label analysis, manual high-value sample selection, lightweight annotation tooling, and few-shot target-domain retraining. With 50 manually refined target-domain images and 479 held-out evaluation images, the final YOLO11s model achieves `mAP50=0.9877` and `mAP50-95=0.7183`.

Additional documents:

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md): portfolio summary.
- [REPORT.md](REPORT.md): Chinese course-style report.
- [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md): ablations and visualizations.
