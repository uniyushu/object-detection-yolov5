#!/bin/bash

PRUNE_RATIO=${1:-0.5}
IMG_SIZE=${2:-416}
MODEL=${3:-yolov5l}
BATCH_SIZE=${4:-64}
SPARSITY_TYPE=${5:-block_punched}
SP_CONFIG_FILE=${6:-./prune_profiles/yolov5l_${PRUNE_RATIO}.yaml}
GPU=${7:-0,1,2,3}
HYP=${8:-./data/hyps/hyp.scratch-high.yaml}
EPOCHS=${9:-300}
SP_ADMM_BLOCK=${10:-8,4}
PORT=$(($RANDOM % 1000 + 1000))

python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node 4 train.py \
    --batch-size ${BATCH_SIZE} \
    --data coco.yaml \
    --hyp ${HYP} \
    --img-size ${IMG_SIZE} \
    --weights ./runs/train/exp97/weights/last.pt  \
    --device ${GPU} \
    --cfg ${MODEL}.yaml \
    --sp-retrain \
    --sp-prune-before-retrain \
    --sp-admm-sparsity-type ${SPARSITY_TYPE} \
    --sp-config-file ${SP_CONFIG_FILE} \
    --sp-admm-block ${SP_ADMM_BLOCK} \
    --epochs ${EPOCHS}
