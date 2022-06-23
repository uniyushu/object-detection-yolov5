#!/bin/bash

EVOLVE=${1:-0}
ADMM_LR=${2:-0.01}
MODEL=${3:-yolov5_30g_8_4}
BATCH_SIZE=${4:-64}
SPARSITY_TYPE=${5:-block_punched}
SP_CONFIG_FILE=${6:-./prune_profiles/yolov5_30g_8_4_0.75.yaml}
GPU=${7:-0,1,2,3}
HYP=${8:-./data/hyps/hyp.scratch-high.yaml}
EPOCHS=${9:-75}
SP_ADMM_BLOCK=${10:-8,4}
IMG_SIZE=${11:-640}
PORT=$(($RANDOM % 1000 + 1000))

echo $EVOLVE
if [ ${EVOLVE} -eq 1 ]; then
    echo 'evolve hyperparameters'
    python train.py \
        --batch-size ${BATCH_SIZE} \
        --data coco.yaml \
        --hyp ${HYP} \
        --weights ./${MODEL}.pt \
        --cfg ${MODEL}.yaml \
        --img-size ${IMG_SIZE} \
        --sp-admm \
        --sp-admm-sparsity-type ${SPARSITY_TYPE} \
        --sp-admm-lr ${ADMM_LR} \
        --epochs ${EPOCHS} \
        --sp-admm-update-epoch $((${EPOCHS} / 6)) \
        --sp-admm-block ${SP_ADMM_BLOCK} \
        --sp-config-file ${SP_CONFIG_FILE} \
        --evolve
else
    echo 'not evolve hyperparameters'
    python -m torch.distributed.launch --master_port ${PORT} --nproc_per_node 4 train.py \
        --batch-size ${BATCH_SIZE} \
        --data coco.yaml \
        --hyp ${HYP} \
        --weights ./runs/train/exp29/weights/best.pt \
        --device ${GPU} \
        --cfg ${MODEL}.yaml \
        --img-size ${IMG_SIZE} \
        --sp-admm \
        --sp-admm-sparsity-type ${SPARSITY_TYPE} \
        --sp-admm-lr ${ADMM_LR} \
        --epochs ${EPOCHS} \
        --sp-admm-update-epoch $((${EPOCHS} / 6)) \
        --sp-admm-block ${SP_ADMM_BLOCK} \
        --sp-config-file ${SP_CONFIG_FILE} \
        --admm-debug
fi


