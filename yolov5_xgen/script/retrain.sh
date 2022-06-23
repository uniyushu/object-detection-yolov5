#!/bin/bash

EVOLVE=${1:-0}
MODEL=${2:-yolov5_30g_8_4}
BATCH_SIZE=${3:-64}
SPARSITY_TYPE=${4:-block_punched}
SP_CONFIG_FILE=${5:-./prune_profiles/yolov5_30g_8_4_0.75.yaml}
GPU=${6:-0,1,2,3}
# HYP=${7:-./data/hyps/hyp.scratch.yaml}
HYP=${7:-./data/hyps/hyp.scratch-high.yaml}
IMG_SIZE=${8:-640}
EPOCHS=${9:-300}
SP_ADMM_BLOCK=${10:-8,4}
PORT=$(($RANDOM % 1000 + 1000))

echo $EVOLVE
if [ ${EVOLVE} -eq 1 ]; then
    echo 'evolve hyperparameters'
    for i in 0 1 2 3; do
        python train.py \
            --batch-size ${BATCH_SIZE} \
            --data coco.yaml \
            --hyp ${HYP} \
            --img-size ${IMG_SIZE} \
            --weights ./runs/train/exp37/weights/last.pt \
            --cfg ${MODEL}.yaml \
            --sp-retrain --sp-prune-before-retrain \
            --sp-admm-sparsity-type ${SPARSITY_TYPE} \
            --sp-config-file ${SP_CONFIG_FILE} \
            --device $i \
            --evolve 2>&1 | tee evolve_gpu_$i.log
    done
else
    echo 'not evolve hyperparameters'
    python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node 4 train.py \
        --batch-size ${BATCH_SIZE} \
        --data coco.yaml \
        --hyp ${HYP} \
        --img-size ${IMG_SIZE} \
        --weights ./runs/train/exp42/weights/last.pt \
        --device ${GPU} \
        --cfg ${MODEL}.yaml \
        --sp-retrain --sp-prune-before-retrain \
        --sp-admm-sparsity-type ${SPARSITY_TYPE} \
        --sp-config-file ${SP_CONFIG_FILE} \
        --sp-admm-block ${SP_ADMM_BLOCK}
fi