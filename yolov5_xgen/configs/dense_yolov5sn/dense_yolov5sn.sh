#!/bin/bash

PORT=$(($RANDOM % 1000 + 1000))

python -m torch.distributed.launch \
    --master_port ${PORT} \
    --nproc_per_node 4 train_script_main.py \
    --config configs/dense_yolov5s/dense_yolov5s.json
