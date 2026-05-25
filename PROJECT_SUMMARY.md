# 项目复盘：SCNT 显微目标检测

这个项目围绕 SCNT 显微操作图像中的三类目标检测展开：

| ID | 类别 | 含义 |
| --- | --- | --- |
| 0 | `injection_needle` | 注射针 |
| 1 | `holding_needle` | 吸持针 |
| 2 | `oocyte` | 卵细胞 |

最初的设定是源域训练、目标域验证。源域和目标域在颜色、亮度、目标尺度、针的姿态和背景纹理上差异明显，所以单纯把源域模型拿到目标域上用，漏检和误检都比较多。

## 数据使用方式

我把实验过程分成了几个阶段：

| 数据 | 用法 |
| --- | --- |
| `SCNT-Source` | 源域训练数据 |
| `SCNT-Target` | 目标域数据，早期用于验证和误差分析 |
| `manual_train` | 从目标域中选出的 50 张人工精修样本 |
| `final_eval` | 未进入训练的 479 张目标域图像 |
| `SCNT-ManualMix` | 最终训练集，由源域采样、源域增强和 50 张人工精修目标域样本组成 |

最后一版结果不再称为严格无监督域自适应，因为它使用了 50 张目标域人工标注图像。更准确地说，这是一个小样本目标域监督适应实验。

## 实验路线

### 1. 源域 baseline

先用 `SCNT-Source` 训练 YOLOv8n，并在目标域上验证。这个阶段的目的不是追求高分，而是确认源域到目标域的迁移难度。

结果显示，源域模型在目标域上表现较差，尤其是卵细胞和针类目标。

### 2. 源域增强

随后换成 YOLO11s，并加入更贴近显微图像的增强方式，包括颜色变化、亮度变化、尺度变化、小目标构造和轻量几何扰动。

这一阶段明显提升了目标域表现，尤其是卵细胞检测。但针类仍然容易混淆，主要表现为 `holding_needle` 被预测成 `injection_needle`。

### 3. 伪标签自训练

我实现了目标域伪标签生成和二阶段训练流程。实际效果并不稳定，主要原因是教师模型本身会把一部分 holding needle 预测成 injection needle，伪标签会继承这些错误。

这个实验的价值在于确认了一件事：在初始模型错误具有明显类别偏差时，伪标签自训练不一定能解决问题，反而可能放大错误。

### 4. 应用层后处理

针对针类混淆，我试过一个基于框形态的后处理规则，用长宽比、面积和高度筛出一部分更像 holding needle 的预测框。

它能改善部分样例，也能提高自定义评估器里的 holding needle 指标，但规则依赖比较强，所以没有作为最终方案。

### 5. 小样本人工精修

最后我从目标域里选出 50 张有代表性的图像，用模型预标注后人工精修，再和采样后的源域数据、源域增强数据混合训练。

这个阶段的目标是验证：少量高质量目标域标注是否能解决针类混淆和小目标漏检问题。

## 关键结果

| 方法 | 验证集 | AP50 injection | AP50 holding | AP50 oocyte | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Source plain YOLOv8n | target_eval | 0.2474 | 0.2258 | 0.0788 | 0.1840 | 0.0901 |
| YOLO11s + source augmentation | target_eval | 0.6938 | 0.6574 | 0.9608 | 0.7707 | 0.4842 |
| YOLO11s raw reference | full SCNT-Target | 0.6616 | 0.6448 | 0.9499 | 0.7521 | 0.4864 |
| Manual-50 YOLO11s retrain | final_eval 479 | 0.9943 | 0.9774 | 0.9914 | 0.9877 | 0.7183 |

需要注意的是，最后一行使用了 50 张目标域人工标注样本，不能和纯无监督方法直接比较。它更适合作为“少量标注能带来多大收益”的实验结果。

## 主要观察

- 目标尺度很关键。源域里目标普遍更大，目标域里较小的针和卵细胞更容易漏检。
- `holding_needle` 和 `injection_needle` 的视觉差异比想象中小，只依赖源域学习到的外观特征不够稳。
- 伪标签不是万能的。教师模型有系统性错误时，自训练会把错误继续传下去。
- 少量精修目标域样本的收益很明显，尤其能纠正针类的类别边界。
- 后处理能作为工程补丁，但如果希望模型本身稳定，还是需要更好的数据覆盖。

## 工程实现

仓库里保留了完整流程脚本：

| 文件 | 作用 |
| --- | --- |
| `scripts/check_dataset.py` | 检查图片和 YOLO 标签格式 |
| `scripts/train_baseline.py` | 统一训练入口，支持 YOLOv8 / YOLO11 |
| `scripts/val_model.py` | 验证模型并写入 CSV |
| `scripts/build_augmented_source.py` | 构建源域增强数据 |
| `scripts/generate_pseudo_labels.py` | 生成目标域伪标签 |
| `scripts/eval_postprocess.py` | 对比原始预测和后处理预测 |
| `scripts/review_select_images.py` | 快速筛选人工标注图片 |
| `scripts/annotate_yolo.py` | OpenCV YOLO 标注工具 |
| `scripts/build_manual_finetune_dataset.py` | 构建最终混合训练集 |

数据集、训练权重和完整输出没有放进仓库，只保留少量代表性可视化图片。

## 复现入口

完整运行命令放在 [README.md](README.md)。比较重要的几个入口是：

```powershell
python scripts/check_dataset.py
python scripts/train_baseline.py --data configs/scnt_source_aug.yaml --model yolo11s.pt --imgsz 960 --batch 2 --rect --device 0
python scripts/review_select_images.py --target-count 50 --reset-selection --overwrite-export
python scripts/annotate_yolo.py --image-dir dataset/SCNT-ManualSelf/manual_train/images --label-dir dataset/SCNT-ManualSelf/manual_train/labels
python scripts/build_manual_finetune_dataset.py --source-frac 0.20 --aug-per-source 2 --manual-repeat 4 --seed 42 --overwrite
python scripts/val_model.py --model runs/scnt/manual50_mix_yolo11s_from_pretrain_960/weights/best.pt --data configs/scnt_manual_mix.yaml --imgsz 960 --batch 2 --rect --device 0
```

## 还可以继续做的事

- 把人工筛图升级成更标准的 active learning 流程。
- 收集更多设备、颜色和操作阶段下的数据。
- 试 YOLO11m、RT-DETR、DINO 等检测器，比较速度和精度。
- 做一个简单推理界面，用于上传显微图像并查看检测结果。
