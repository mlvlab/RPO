#!/bin/bash

#cd ../..

# custom config
#DATA=/hub_data2/intern/data/
TRAINER=RPO

DATASET=$1
SEED=$2
GPU=$3

CFG=$4

SHOTS=$5
LOADEP=$6
SUB=$7


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/rpo/base2new/train_base/${COMMON_DIR}
DIR=output/rpo/base2new/test_${SUB}/${COMMON_DIR}

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