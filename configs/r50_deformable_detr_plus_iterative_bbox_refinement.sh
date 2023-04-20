#!/usr/bin/env bash

# Default, plus ibboxref (box refinement) mechanism to improve detection performance.
# Each decoder layer refines the previous layer's bounding box predictions.
# At each decoder layer, the new bounding box is computed as a function of the previous box and the box that would be predicted here.

set -x

EXP_DIR=exps/r50_deformable_detr_plus_iterative_bbox_refinement
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    ${PY_ARGS}
