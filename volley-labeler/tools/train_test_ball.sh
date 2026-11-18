yolo detect train \
    data="$1" \
    model=yolo11n.pt \
    imgsz=640 \
    epochs=80 \
    batch=4 \
    device="${2:-0}" \
    workers=2 \
    flipud=0 \
    degrees=0 \
    shear=0 \
    perspective=0 \
    mosaic=0.5 \
    hsv_v=0.4 \
    patience=25 \
    project=runs \
    name=ball
