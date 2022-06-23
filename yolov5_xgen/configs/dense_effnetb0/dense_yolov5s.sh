#!/bin/bash

MODEL=${1:-yolov5l}
BATCH_SIZE=${2:-64}
GPU=${3:-0,1,2,3}
HYP=${4:-./data/hyps/hyp.scratch-high.yaml}
IMG_SIZE=${5:-416}
PORT=$(($RANDOM % 1000 + 1000))

if [[ $MODEL == *"6"* ]]; then
    MODEL=./models/hub/$MODEL
fi

python -m torch.distributed.launch \
    --master_port ${PORT} \
    --nproc_per_node 4 train.py \
    --batch ${BATCH_SIZE} \
    --hyp ${HYP} \
    --data coco.yaml \
    --cfg ${MODEL}.yaml \
    --weights '' \
    --device ${GPU} \
    --img-size ${IMG_SIZE}
