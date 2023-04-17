#!/usr/bin/env bash

# Like the single_scale version, but we dilate the C5 stage of the backbone 
# (see the original DETR paper and https://arxiv.org/abs/1611.07709)

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale_dc5
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --dilation \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
