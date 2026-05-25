# Domain-Adaptive Microscopic Object Detection for SCNT

本项目面向显微操作场景中的目标检测任务，检测对象包括注射针、吸持针和卵细胞。项目从源域监督训练开始，逐步尝试域泛化增强、伪标签自训练、误差驱动人工精标和小样本目标域再训练，最终形成一套比较完整的显微目标检测实验流程。

## 项目亮点

- 使用 Ultralytics YOLO 完成 SCNT 显微图像三类目标检测。
- 严格区分源域标签、目标域无标签图像、目标域最终验证标签。
- 实现数据检查、目标域划分、训练、验证、伪标签生成、可视化、人工筛图和人工标注辅助脚本。
- 针对 holding needle 与 injection needle 混淆、小目标漏检、暖色背景迁移等问题做了系统误差分析。
- 引入 50 张目标域人工精标样本，构建小样本目标域监督微调实验，并在 479 张独立 final_eval 上验证。

## 类别定义

| ID | 类别 | 中文说明 |
| --- | --- | --- |
| 0 | `injection_needle` | 注射针 |
| 1 | `holding_needle` | 吸持针 |
| 2 | `oocyte` | 卵细胞 |

## 数据集结构

原始数据集目录如下：

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

实验中使用的数据规模：

| 数据部分 | 数量 | 用途 |
| --- | ---: | --- |
| SCNT-Source | 2785 | 源域监督训练 |
| SCNT-Target | 529 | 目标域图像 |
| Manual target train | 50 | 人工精标目标域训练样本 |
| Final eval | 479 | 最终验证集 |
| ManualMix train | 1871 | 源域采样 + 增强 + 50 张目标域精标重复采样 |

注意：`SCNT-Target/labels` 只用于最终验证和误差分析，不能直接混入伪标签训练或无监督自训练。

## 环境安装

```powershell
pip install -r requirements.txt
```

推荐使用 GPU 版 PyTorch。当前主要实验环境：

```text
Python 3.12
Ultralytics 8.4.51
torch 2.10.0+cu126
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

## 数据检查

```powershell
python scripts/check_dataset.py
```

输出：

```text
outputs/dataset_check.txt
```

检查内容包括：

- 图像和标签是否一一对应。
- 标签是否为 YOLO 格式：`class x_center y_center width height`。
- 类别是否只包含 `0, 1, 2`。
- 归一化 bbox 坐标是否在 `[0, 1]`。
- 是否存在缺失标签、空标签、非法类别或非法坐标。

## 目标域划分

严格实验可以先做 7:3 划分：

```powershell
python scripts/split_target.py --overwrite
```

后续为了做小样本人工精标实验，本项目采用了另一套人工筛图划分：

```text
dataset/SCNT-ManualSelf/manual_train/images  # 50 张人工精标目标域图像
dataset/SCNT-ManualSelf/manual_train/labels
dataset/SCNT-ManualSelf/final_eval/images    # 479 张最终验证图像
dataset/SCNT-ManualSelf/final_eval/labels
```

## Baseline 训练

源域 baseline：

```powershell
python scripts/train_baseline.py --data configs/scnt_source.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --project runs/scnt --name baseline
```

YOLO11s + 增强实验：

```powershell
python scripts/train_baseline.py --data configs/scnt_source_aug.yaml --model yolo11s.pt --epochs 50 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name source_aug_yolo11s_compact_smallobj_960 --exist-ok --quiet
```

## 验证模型

```powershell
python scripts/val_model.py --model runs/scnt/source_aug_yolo11s_compact_smallobj_960/weights/best.pt --data configs/scnt_source.yaml --name source_aug_yolo11s_compact_smallobj_960 --imgsz 960 --batch 2 --rect --device 0 --summary-csv outputs/experiments_summary.csv --exist-ok --quiet
```

指标会写入：

```text
outputs/experiments_summary.csv
```

## 伪标签自训练

生成伪标签：

```powershell
python scripts/generate_pseudo_labels.py --model runs/scnt/source_aug_yolo11s_compact_smallobj_960/weights/best.pt --overwrite --device 0
```

默认筛选阈值：

```text
class 0 injection_needle: conf >= 0.55
class 1 holding_needle:   conf >= 0.55
class 2 oocyte:           conf >= 0.75
```

构建伪标签训练集：

```powershell
python scripts/build_pseudo_dataset.py --overwrite
```

训练：

```powershell
python scripts/train_pseudo_adapt.py --epochs 50 --imgsz 640 --batch 16
```

实验观察：伪标签自训练容易受到 confirmation bias 影响，尤其在 holding needle 与 injection needle 混淆明显时，错误伪标签会放大模型偏差。因此本项目最终采用人工精标小样本监督适应作为主结果。

## 人工筛图与标注

快速查看模型原始预测并手动选择高价值样本：

```powershell
python scripts/review_select_images.py --target-count 50 --reset-selection --overwrite-export
```

按键：

```text
空格 / y：选中
n / d：跳过
p / a：上一张
u / x / Delete：取消选中
s：保存当前选择
q / Esc：保存并退出
```

对选中的图片生成预标注：

```powershell
python scripts/prelabel_fewshot.py --model runs/scnt/source_aug_yolo11s_compact_smallobj_960/weights/best.pt --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels --imgsz 960 --device 0 --predict-conf 0.25 --conf-class-0 0.25 --conf-class-1 0.25 --conf-class-2 0.25 --overwrite-labels
```

人工精修：

```powershell
python scripts/annotate_yolo.py --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels
```

标注工具按键：

```text
0：injection_needle
1：holding_needle
2：oocyte
鼠标拖拽：新增框
单击框：选中框
Delete / 右键：删除框
s：保存当前图
n：保存并下一张
p：保存并上一张
q / Esc：保存并退出
```

## 构建最终小样本训练集

本项目最终使用：

- 源域原图随机 20%：557 张。
- 每张源域图随机选 2 张离线增强图：1114 张。
- 人工精标目标域 50 张，重复采样 4 次：200 张。

构建命令：

```powershell
python scripts/build_manual_finetune_dataset.py --source-frac 0.20 --aug-per-source 2 --manual-repeat 4 --seed 42 --overwrite
```

输出：

```text
dataset/SCNT-ManualMix/
configs/scnt_manual_mix.yaml
```

## 最终训练命令

从 COCO 预训练的 YOLO11s 重新训练，而不是从旧 SCNT 模型微调：

```powershell
python scripts/train_baseline.py --data configs/scnt_manual_mix.yaml --model yolo11s.pt --epochs 80 --imgsz 960 --batch 2 --rect --device 0 --workers 2 --project runs/scnt --name manual50_mix_yolo11s_from_pretrain_960 --patience 20 --hsv-h 0.005 --hsv-s 0.2 --hsv-v 0.15 --scale 0.15 --translate 0.05 --mosaic 0.2 --mixup 0.0 --fliplr 0.0 --flipud 0.0 --close-mosaic 15 --exist-ok --quiet
```

实际训练中第 15 轮达到最佳，后续指标下降，因此提前停止。最终采用：

```text
runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt
```

## 测试集可视化

```powershell
python scripts/predict_visualize.py --model runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt --source dataset/SCNT-ManualSelf/final_eval/images --output outputs/visualizations/manual50_mix_yolo11s_from_pretrain_960_final_eval_raw --max-images 0 --conf 0.25 --imgsz 960 --batch 1 --rect --device 0 --exist-ok --quiet
```

打开可视化结果：

```powershell
explorer outputs\visualizations\manual50_mix_yolo11s_from_pretrain_960_final_eval_raw
```

## 实验结果

| 实验 | 验证集 | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Source plain YOLOv8n | target_eval | 0.2474 | 0.2258 | 0.0788 | 0.1840 | 0.0901 |
| YOLO11s + source augmentation | target_eval | 0.6938 | 0.6574 | 0.9608 | 0.7707 | 0.4842 |
| YOLO11s raw reference | full SCNT-Target | 0.6616 | 0.6448 | 0.9499 | 0.7521 | 0.4864 |
| Manual-50 YOLO11s retrain | final_eval 479 | 0.9943 | 0.9774 | 0.9914 | 0.9877 | 0.7183 |

更多探索实验结果见：

```text
docs/EXPERIMENT_RESULTS.md
```

其中保留了“YOLO11s + 源域图像增强 + 应用层形态后处理”的结果。该后处理实验在 full SCNT-Target 自定义评估中将 holding needle AP50 从 `0.0784` 提升到 `0.4987`，mAP50 从 `0.4611` 提升到 `0.6234`。这部分不是最终主方法，但能说明 holding/injection 混淆可以通过形态约束被部分纠正。

说明：最后一行使用了 50 张目标域人工精标样本，因此不再属于严格无监督域自适应，而是“小样本目标域监督适应 / few-shot supervised domain adaptation”。

## 主要结论

1. 仅源域训练在目标域上性能较差，说明源域和目标域存在明显 domain gap。
2. 图像增强和更强的 YOLO11s 模型能显著提升 oocyte 检测，但 holding needle 与 injection needle 仍容易混淆。
3. 伪标签自训练在目标域误检较多时会受到 confirmation bias 影响，不一定稳定提升。
4. 通过错误分析筛选少量目标域样本进行人工精标，再与源域采样和增强数据混合训练，是本项目中最有效的改进路线。
5. 最终模型在 479 张独立 final_eval 上达到 `mAP50=0.9877`、`mAP50-95=0.7183`。

## GitHub 说明

本仓库默认不上传以下内容：

```text
dataset/
runs/
outputs/
*.pt
```

原因是数据集、训练权重和可视化结果通常体积较大，也可能涉及课程或数据授权限制。完整复现实验需要在本地放置数据集，并按 README 中的命令重新生成中间文件。
