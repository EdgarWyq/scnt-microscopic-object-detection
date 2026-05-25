# 域自适应显微目标检测课程项目报告

## 1. 项目背景

本项目研究 SCNT 显微操作图像中的目标检测问题，目标是在显微视野中检测注射针、吸持针和卵细胞。该任务具有明显的工程难点：显微图像分辨率有限、背景纹理复杂、针体细长且边缘弱、卵细胞可能失焦或被截断，不同显微设备和成像条件还会造成颜色、亮度、对比度和尺度差异。

项目最初设定为源域到目标域的域自适应检测任务：源域 `SCNT-Source` 具有真实标签，可用于训练；目标域 `SCNT-Target` 的真实标签原则上只用于最终验证。实验过程中先后实现了源域 baseline、显微图像增强、伪标签无监督自训练和少量目标域人工精标监督适应。

## 2. 数据集介绍

原始数据集结构如下：

```text
dataset/
└── SCNT/
    ├── SCNT-Source/
    │   ├── images/
    │   └── labels/
    └── SCNT-Target/
        ├── images/
        └── labels/
```

检测类别如下：

| ID | 类别 | 中文说明 |
| --- | --- | --- |
| 0 | `injection_needle` | 注射针 |
| 1 | `holding_needle` | 吸持针 |
| 2 | `oocyte` | 卵细胞 |

本项目中统计到的数据规模：

| 数据部分 | 图像数 | 用途 |
| --- | ---: | --- |
| SCNT-Source | 2785 | 源域训练 |
| SCNT-Target | 529 | 目标域图像 |
| Manual target train | 50 | 人工精标目标域训练样本 |
| Final eval | 479 | 最终验证集 |
| ManualMix train | 1871 | 源域采样、增强和人工目标域样本混合训练 |

## 3. 数据检查

实现脚本：

```text
scripts/check_dataset.py
```

检查内容包括：

- 图像和标签是否一一对应。
- YOLO 标签是否为 `class x_center y_center width height`。
- 类别编号是否只包含 `0, 1, 2`。
- bbox 坐标是否在 `[0, 1]` 范围内。
- 是否存在缺失标签、空标签、非法类别和非法坐标。

检查结果输出到：

```text
outputs/dataset_check.txt
```

## 4. 源域与目标域差异分析

从训练和可视化结果看，源域和目标域差异主要体现在：

1. 目标尺度差异：源域中三类目标普遍较大，目标域中存在更多小针、小卵细胞和远景视野。
2. 颜色差异：目标域中存在明显暖色、橙色或偏红背景，源域灰度图像较多。
3. 形态差异：holding needle 和 injection needle 都可能呈现细长结构，且方向、宽度、尖端形态变化较大。
4. 成像差异：部分图像存在模糊、低对比度、边缘截断和杂质小黑点。
5. 类间混淆：模型容易将横向或粗大的 holding needle 预测为 injection needle。

这些差异导致纯源域训练模型在目标域上出现漏检、误检和类别混淆。

## 5. 方法一：源域 YOLO Baseline

Baseline 使用 Ultralytics YOLO 进行源域监督训练，训练数据仅来自 `SCNT-Source/images` 和对应 labels，目标域只用于验证。

默认实现脚本：

```text
scripts/train_baseline.py
scripts/val_model.py
```

源域 plain baseline 的目标域表现较差，说明源域和目标域之间存在明显 domain gap。

## 6. 方法二：显微图像增强的域泛化

为提高模型对显微成像差异的鲁棒性，项目加入了适合显微图像的增强策略：

- HSV / brightness / value 扰动。
- scale 和 translate。
- mosaic，默认降低到较温和的比例。
- mixup 保持较低或关闭。
- fliplr 设为 0，因为显微操作图像左右结构可能具有实际语义。
- 离线增强包括颜色风格变化、橙色背景模拟、缩放和小目标增强等。

相关脚本：

```text
scripts/build_augmented_source.py
configs/scnt_source_aug.yaml
```

增强后，YOLO11s 在目标域上的 oocyte 检测显著提升，但针类尤其是 holding needle 与 injection needle 的混淆仍然存在。

## 7. 方法三：基于高置信度伪标签的无监督域自适应

伪标签阶段只读取目标域图像，不读取目标域真实标签。流程如下：

1. 使用 baseline 或增强模型预测目标域图像。
2. 按类别置信度阈值筛选预测框。
3. 删除异常框，例如宽高不合法、过大或面积过小的框。
4. 将源域真实标签和目标域伪标签合并训练。

相关脚本：

```text
scripts/generate_pseudo_labels.py
scripts/build_pseudo_dataset.py
scripts/train_pseudo_adapt.py
```

实验观察：伪标签自训练对初始模型质量非常敏感。当 baseline 已经存在 holding/injection 系统性混淆时，伪标签会继承甚至放大错误，导致 confirmation bias。因此伪标签实验没有作为最终最优方案。

## 8. 方法四：误差驱动的少量目标域人工精标

在伪标签效果不稳定后，项目转向更工程化的数据闭环：

1. 使用当前最优模型对全目标域图像做 raw prediction。
2. 人工快速筛图，优先选择 holding/injection 混淆、小目标漏检、暖色背景、模糊和边缘截断图像。
3. 精修 50 张目标域图像标签。
4. 保留剩余 479 张作为 final_eval，不参与训练。
5. 构建混合训练集：源域随机 20%、对应增强图每张 2 张、50 张人工目标域样本重复采样 4 次。
6. 从 `yolo11s.pt` 重新训练，而不是从旧 SCNT 模型微调。

相关脚本：

```text
scripts/review_select_images.py
scripts/prelabel_fewshot.py
scripts/annotate_yolo.py
scripts/build_manual_finetune_dataset.py
```

这种方法不再是严格无监督域自适应，而是 few-shot supervised domain adaptation。它更符合实际工程中的数据闭环：模型预测、错误分析、少量高价值标注、重新训练和大验证集评估。

## 9. 实验设置

最终实验配置：

| 项目 | 设置 |
| --- | --- |
| 模型 | YOLO11s |
| 输入尺寸 | 960 |
| batch | 2 |
| 训练集 | `configs/scnt_manual_mix.yaml` |
| 训练图像 | 1871 |
| 人工目标域图像 | 50，重复采样 4 次 |
| 验证集 | 479 张 final_eval |
| 训练轮数 | 计划 80，实际最佳 epoch 15，后续提前停止 |
| 设备 | RTX 4060 Laptop GPU |

最终训练命令：

```powershell
python scripts/train_baseline.py --data configs/scnt_manual_mix.yaml --model yolo11s.pt --epochs 80 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name manual50_mix_yolo11s_from_pretrain_960 --patience 20 --hsv-h 0.005 --hsv-s 0.2 --hsv-v 0.15 --scale 0.15 --translate 0.05 --mosaic 0.2 --mixup 0.0 --fliplr 0.0 --flipud 0.0 --close-mosaic 15 --exist-ok --quiet
```

## 10. 评价指标

- AP@0.5：单类别在 IoU=0.5 阈值下的 Average Precision。
- mAP@0.5：所有类别 AP@0.5 的平均值，反映较宽松定位阈值下的检测能力。
- mAP@[0.5:0.95]：IoU 从 0.5 到 0.95 多个阈值下的平均 mAP，对定位精度要求更高。

## 11. 实验结果

| 实验 | 验证集 | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Source plain YOLOv8n | target_eval | 0.2474 | 0.2258 | 0.0788 | 0.1840 | 0.0901 |
| Pseudo adapt | target_eval | 0.1255 | 0.8433 | 0.2445 | 0.4045 | 0.3209 |
| YOLO11s + source augmentation | target_eval | 0.6938 | 0.6574 | 0.9608 | 0.7707 | 0.4842 |
| YOLO11s raw reference | full SCNT-Target | 0.6616 | 0.6448 | 0.9499 | 0.7521 | 0.4864 |
| Manual-50 YOLO11s retrain | final_eval 479 | 0.9943 | 0.9774 | 0.9914 | 0.9877 | 0.7183 |

最终模型路径：

```text
runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt
```

最终可视化结果：

```text
outputs/visualizations/manual50_mix_yolo11s_from_pretrain_960_final_eval_raw
```

## 12. 误检漏检分析

主要错误类型包括：

1. holding needle 被识别为 injection needle：两类都具有细长形态，源域中 holding 的方向和形态分布较单一，导致模型学习到偏置。
2. 小目标漏检：目标域中存在更小、更远、更模糊的针和卵细胞。
3. 小黑点误检为 oocyte：显微图中杂质、气泡和碎片容易产生类似圆形暗斑。
4. 边缘截断目标：卵细胞或针体只露出部分区域时，模型定位和分类更不稳定。
5. 暖色背景迁移：橙色或红色背景对只在灰度源域训练的模型有明显影响。

人工精标阶段重点针对这些错误类型选图和修正。

## 13. 个人贡献

- 搭建完整 YOLO 检测实验工程。
- 编写数据检查、目标域划分、训练、验证、可视化、伪标签和结果对比脚本。
- 设计显微图像增强实验和离线增强数据集。
- 分析 holding/injection 混淆、小目标漏检和颜色迁移问题。
- 实现人工筛图和 OpenCV YOLO 标注工具。
- 构建 50 张目标域人工精标集，并完成小样本监督适应实验。
- 在 479 张独立 final_eval 上完成最终评估和可视化。

## 14. 结论

本项目表明，单纯源域训练难以直接泛化到目标显微域；图像增强和更强 YOLO 模型能显著提升性能，但仍无法完全解决针类混淆。伪标签自训练在初始模型误差较大时容易受到 confirmation bias 影响。相比之下，基于错误分析的少量目标域人工精标能高效提升模型在目标域上的表现。最终模型在 479 张独立验证图像上达到 `mAP50=0.9877` 和 `mAP50-95=0.7183`，验证了“小样本精标 + 源域采样增强 + 重新训练”的有效性。
