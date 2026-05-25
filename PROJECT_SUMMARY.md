# 项目总结：SCNT 显微目标检测

## 一句话介绍

基于 Ultralytics YOLO 构建显微操作图像目标检测系统，检测注射针、吸持针和卵细胞；通过域泛化增强、伪标签自训练和少量目标域人工精标，最终在 479 张目标域验证图像上达到 `mAP50=0.9877`、`mAP50-95=0.7183`。

## 技术栈

- Python
- Ultralytics YOLO / YOLO11s
- PyTorch
- OpenCV
- pandas / matplotlib / tqdm
- Windows + PyCharm + RTX 4060 Laptop GPU

## 我解决的问题

源域和目标域显微图像存在明显差异：目标尺度、颜色背景、针体方向、模糊程度和杂质分布都不同。纯源域训练模型在目标域上容易漏检小目标，也容易把 holding needle 和 injection needle 混淆。

项目最终采用错误驱动的数据闭环：

1. 训练源域 baseline。
2. 增加显微图像域泛化增强。
3. 尝试伪标签无监督自训练并分析 confirmation bias。
4. 人工筛选 50 张高价值目标域图像进行精标。
5. 使用源域 20% 采样、每张 2 个增强版本和 50 张人工目标域样本混合训练。
6. 在剩余 479 张目标域图像上做最终验证。

## 最终结果

| 方法 | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source plain YOLOv8n | 0.2474 | 0.2258 | 0.0788 | 0.1840 | 0.0901 |
| YOLO11s + source augmentation | 0.6938 | 0.6574 | 0.9608 | 0.7707 | 0.4842 |
| Manual-50 YOLO11s retrain | 0.9943 | 0.9774 | 0.9914 | 0.9877 | 0.7183 |

## 工程产出

- `scripts/check_dataset.py`：YOLO 数据集合法性检查。
- `scripts/train_baseline.py`：统一 YOLO 训练入口。
- `scripts/val_model.py`：模型验证并写入 CSV。
- `scripts/generate_pseudo_labels.py`：目标域伪标签生成。
- `scripts/build_augmented_source.py`：离线显微图像增强。
- `scripts/review_select_images.py`：人工快速筛图。
- `scripts/annotate_yolo.py`：轻量 OpenCV YOLO 标注工具。
- `scripts/build_manual_finetune_dataset.py`：构建最终小样本目标域适应训练集。
- `scripts/predict_visualize.py`：批量预测可视化。

## 项目价值

这个项目不只是跑通 YOLO，而是完整覆盖了真实视觉项目中常见的数据闭环：

- 数据检查。
- Baseline 建立。
- 误差分析。
- 增强实验。
- 伪标签方案验证。
- 人工高价值样本筛选。
- 小样本精标。
- 重新训练和独立验证。
- 可视化结果复查。

## 可进一步改进

- 用更标准的 active learning 策略自动推荐待标注样本。
- 引入更大规模的显微操作数据集，提高泛化能力。
- 进一步分析 holding/injection 的形态特征，设计更稳定的类别判别策略。
- 尝试 YOLO11m、RT-DETR、DINO 系列检测器。
- 增加模型导出与实时推理界面。
