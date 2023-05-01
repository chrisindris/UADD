#!/usr/bin/env bash

set -x

EXP_DIR=exps/eca_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --eca \
    --cache_mode
    ${PY_ARGS}