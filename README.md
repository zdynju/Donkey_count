# Donkey_count
我现在需要设计一个农场驴目标检测算法，输入是农场俯视视角的监控图片，要求输出目标检测框和返回时间段中画面内的驴的数量,请制定计划，保存在plan.md中

## 当前工程状态

本项目已先搭建目标检测与计数 pipeline。即使暂时没有数据，也可以先完成代码结构、训练入口、图片推理、视频时间段统计和评估工具，后续补充数据和模型权重后即可运行。

## 目录结构

```text
Donkey_count/
├── configs/train.yaml          # 训练配置
├── data/dataset.yaml           # YOLO 数据集配置
├── scripts/
│   ├── extract_frames.py       # 视频抽帧
│   ├── train.py                # 模型训练入口
│   ├── infer_image.py          # 图片检测与计数
│   ├── infer_video.py          # 视频时间段检测与计数
│   └── evaluate_count.py       # 计数误差评估
├── src/
│   ├── detector.py             # YOLO 检测器封装
│   ├── counter.py              # 单帧和时间段计数统计
│   ├── video_io.py             # 视频读取和时间解析
│   └── visualization.py        # 检测框可视化
├── outputs/                    # 推理、训练和报告输出
├── models/                     # 模型权重目录
├── tests/                      # 基础单元测试
└── plan.md                     # 项目实施计划
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

从监控视频中抽帧：

```bash
python scripts/extract_frames.py \
  --source data/raw/farm.mp4 \
  --output-dir data/raw/frames \
  --sample-fps 1
```

标注完成后，按 YOLO 格式组织数据：

```text
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

类别只有一个：`donkey`。

## 训练

```bash
python scripts/train.py --config configs/train.yaml
```

训练完成后，将最佳权重复制或软链接到：

```text
models/best.pt
```

## 图片推理

```bash
python scripts/infer_image.py \
  --source data/raw/example.jpg \
  --weights models/best.pt \
  --conf 0.25
```

输出：

- `outputs/images/*_detected.jpg`
- `outputs/images/*_detections.json`

## 视频时间段计数

```bash
python scripts/infer_video.py \
  --source data/raw/farm.mp4 \
  --weights models/best.pt \
  --start 00:01:00 \
  --end 00:03:00 \
  --sample-fps 2 \
  --save-video
```

输出：

- `outputs/reports/*_count_report.json`
- `outputs/reports/*_count_by_time.csv`
- `outputs/videos/*_detected.mp4`

## 计数评估

如果已有人工统计的真值 CSV，可以运行：

```bash
python scripts/evaluate_count.py \
  --pred outputs/reports/farm_count_by_time.csv \
  --truth data/ground_truth/farm_count_by_time.csv
```

CSV 至少需要包含：

```csv
frame_index,count
1,3
2,4
```
