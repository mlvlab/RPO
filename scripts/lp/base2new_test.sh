#!/bin/bash

# custom config
#DATA=/hub_data2/intern/data/
TRAINER=LP

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1
SHOTS=$3
LOADEP=30
SUB=$4
GPU=$5


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/lp/base2new/train_base/${COMMON_DIR}
DIR=output/lp/base2new/test_${SUB}/${COMMON_DIR}
#if [ -d "$DIR" ]; then
#    echo "Oops! The results exist at ${DIR} (so skip this job)"
#else
CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
#fi