#!/usr/bin/env bash
# usage: tools/train_ball.sh export/dataset_XXXX/ball_yolo/data.yaml [device]
# Small ball at 720p-1080p: train at full res. Camera is fixed => no flips/rotations/perspective warps.
yolo detect train data="$1" model=yolo11s.pt imgsz=1280 epochs=80 batch=1 device="${2:-0}" \
     flipud=0 degrees=0 shear=0 perspective=0 mosaic=0.5 hsv_v=0.4 patience=25 project=runs name=ball
echo "weights: runs/ball/weights/best.pt  ->  python -m volleyscout analyze ... --ball-weights runs/ball/weights/best.pt"
