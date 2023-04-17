#!/usr/bin/env bash

# unlike the default r50_deformable_detr.sh, 
# this uses the single-scale deformable attention module, 
# where only one feature level is used (the default uses 4).

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
