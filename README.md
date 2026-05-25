# SCNT 显微目标检测

这个仓库记录了一个显微操作图像目标检测项目。任务是从 SCNT 显微图像中检测三类目标：注射针、吸持针和卵细胞。

项目一开始按照“源域训练、目标域验证”的域自适应任务来做。后面在实验中发现，单纯依赖源域增强和伪标签很难稳定解决 holding needle / injection needle 的混淆，所以又补充了少量目标域人工精标样本，做了一版小样本目标域适应实验。

数据集、训练权重和完整输出结果没有放在仓库里，只保留代码、配置、报告和少量可视化样例。

## 任务类别

| ID | 类别 | 说明 |
| --- | --- | --- |
| 0 | `injection_needle` | 注射针 |
| 1 | `holding_needle` | 吸持针 |
| 2 | `oocyte` | 卵细胞 |

## 数据划分

原始目录：

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

本项目最后使用的主要划分：

| 数据 | 数量 | 用途 |
| --- | ---: | --- |
| SCNT-Source | 2785 | 源域训练 |
| SCNT-Target | 529 | 目标域数据 |
| manual_train | 50 | 从目标域中筛选并人工精修 |
| final_eval | 479 | 未参与训练的最终验证集 |
| SCNT-ManualMix | 1871 | 源域采样、源域增强和 manual_train 组成的训练集 |

说明：最终结果使用了 50 张目标域人工标注样本，因此更准确地说是“小样本目标域监督适应”，不是严格无监督域自适应。

## 结果

| 实验 | 验证集 | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Source plain YOLOv8n | target_eval | 0.2474 | 0.2258 | 0.0788 | 0.1840 | 0.0901 |
| YOLO11s + source augmentation | target_eval | 0.6938 | 0.6574 | 0.9608 | 0.7707 | 0.4842 |
| YOLO11s raw reference | full SCNT-Target | 0.6616 | 0.6448 | 0.9499 | 0.7521 | 0.4864 |
| Manual-50 YOLO11s retrain | final_eval 479 | 0.9943 | 0.9774 | 0.9914 | 0.9877 | 0.7183 |

最后一行是在当前 50/479 划分上的结果。它说明少量高质量目标域标注对这个任务帮助很明显，但不应和纯无监督域自适应结果混在一起比较。

## 可视化样例

### 最终模型

| 示例 1 | 示例 2 | 示例 3 |
| --- | --- | --- |
| ![](docs/assets/representative_results/manual50_final/1293.jpg) | ![](docs/assets/representative_results/manual50_final/1470.jpg) | ![](docs/assets/representative_results/manual50_final/1501.jpg) |

### 源域增强模型：原始预测和后处理对比

| 原始预测 | 后处理 |
| --- | --- |
| ![](docs/assets/representative_results/source_aug_raw/1293.jpg) | ![](docs/assets/representative_results/source_aug_postprocess/1293.jpg) |
| ![](docs/assets/representative_results/source_aug_raw/1649.jpg) | ![](docs/assets/representative_results/source_aug_postprocess/1649.jpg) |

更多图和消融记录在 [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)。

## 做过的实验

### 1. 源域 baseline

先用 `SCNT-Source` 训练 YOLOv8n，并在目标域上验证。结果较低，说明源域和目标域之间有明显差异。

### 2. 源域增强

使用 YOLO11s，并加入显微图像相关增强。增强主要针对颜色、尺度、小目标和背景变化。这个阶段对卵细胞检测提升明显，但针类混淆仍然比较明显。

### 3. 伪标签自训练

实现了目标域伪标签生成和二阶段训练。实际效果不稳定，主要问题是初始模型会把一部分 holding needle 预测成 injection needle，伪标签会把这种错误继续带进训练。

### 4. 应用层后处理

针对针类混淆，尝试过一个简单的形态后处理：根据预测框的高度、长宽比和面积，把一部分疑似 holding needle 的 injection 预测重标为 holding。

在 full SCNT-Target 上，自定义 evaluator 的结果如下：

| Mode | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.3904 | 0.0784 | 0.9145 | 0.4611 | 0.3119 |
| postprocess | 0.4592 | 0.4987 | 0.9122 | 0.6234 | 0.4351 |

这个实验有参考价值，但它依赖规则，不是最终方案。

### 5. 少量目标域人工精标

最后用模型预测结果辅助筛图，挑了 50 张目标域图像人工精修。训练集由三部分组成：

- 源域原图随机 20%。
- 每张源域图随机选 2 张增强图。
- 50 张目标域精标图重复采样 4 次。

模型从 `yolo11s.pt` 重新训练，最佳结果出现在约第 15 轮。

## 代码结构

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
```

几个主要脚本：

| 脚本 | 作用 |
| --- | --- |
| `scripts/check_dataset.py` | 检查图片和 YOLO 标签是否合法 |
| `scripts/train_baseline.py` | YOLO 训练入口 |
| `scripts/val_model.py` | 验证模型并写入 CSV |
| `scripts/build_augmented_source.py` | 构建源域增强数据 |
| `scripts/generate_pseudo_labels.py` | 生成目标域伪标签 |
| `scripts/eval_postprocess.py` | 对比 raw prediction 和后处理 |
| `scripts/review_select_images.py` | 快速筛选要人工标注的图片 |
| `scripts/annotate_yolo.py` | OpenCV 标注工具 |
| `scripts/build_manual_finetune_dataset.py` | 构建最终混合训练集 |

## 运行方式

安装依赖：

```powershell
pip install -r requirements.txt
```

检查数据：

```powershell
python scripts/check_dataset.py
```

训练源域增强模型：

```powershell
python scripts/train_baseline.py --data configs/scnt_source_aug.yaml --model yolo11s.pt --epochs 50 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name source_aug_yolo11s_compact_smallobj_960 --exist-ok --quiet
```

筛选目标域人工精标样本：

```powershell
python scripts/review_select_images.py --target-count 50 --reset-selection --overwrite-export
```

生成预标注：

```powershell
python scripts/prelabel_fewshot.py --model runs/scnt/source_aug_yolo11s_compact_smallobj_960/weights/best.pt --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels --imgsz 960 --device 0 --predict-conf 0.25 --conf-class-0 0.25 --conf-class-1 0.25 --conf-class-2 0.25 --overwrite-labels
```

人工修正标签：

```powershell
python scripts/annotate_yolo.py --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels
```

构建最终训练集：

```powershell
python scripts/build_manual_finetune_dataset.py --source-frac 0.20 --aug-per-source 2 --manual-repeat 4 --seed 42 --overwrite
```

训练最终模型：

```powershell
python scripts/train_baseline.py --data configs/scnt_manual_mix.yaml --model yolo11s.pt --epochs 80 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name manual50_mix_yolo11s_from_pretrain_960 --patience 20 --hsv-h 0.005 --hsv-s 0.2 --hsv-v 0.15 --scale 0.15 --translate 0.05 --mosaic 0.2 --mixup 0.0 --fliplr 0.0 --flipud 0.0 --close-mosaic 15 --exist-ok --quiet
```

验证：

```powershell
python scripts/val_model.py --model runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt --data configs/scnt_manual_mix.yaml --name manual50_mix_yolo11s_from_pretrain_960_earlystop_best --imgsz 960 --batch 2 --rect --device 0 --summary-csv outputs/experiments_summary.csv --exist-ok --quiet
```

预测可视化：

```powershell
python scripts/predict_visualize.py --model runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt --source dataset/SCNT-ManualSelf/final_eval/images --output outputs/visualizations/manual50_mix_yolo11s_from_pretrain_960_final_eval_raw --max-images 0 --conf 0.25 --imgsz 960 --batch 1 --rect --device 0 --exist-ok --quiet
```

## 没有上传的内容

```text
dataset/
runs/
outputs/
*.pt
```

这些文件体积较大，也可能涉及数据授权。仓库里只保留少量代表性可视化图。

## 补充文档

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)：项目过程复盘。
- [REPORT.md](REPORT.md)：课程报告。
- [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)：消融实验和代表性图片。
