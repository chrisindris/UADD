#!/usr/bin/env bash

# Like the single_scale version, but we dilate the C5 stage of the backbone 
# (see the original DETR paper and https://arxiv.org/abs/1611.07709)
#
# This increases the feature resolution (dilate last stage of backbone and remove a stride from first convolutional block (ResNet has conv blocks)).
# - removing a stride: the stride reduces the amount of the image that the conv filter sees.
# This modifcation increases the resolution by a factor of two, thus improving performance for small
# objects, at the cost of a 16x higher cost in the self-attentions of the encoder,
# leading to an overall 2x increase in computational cost.

set -x

EXP_DIR=exps/r50_deformable_detr_single_scale_dc5
PY_ARGS=${@:1}

python -u main.py \
    --num_feature_levels 1 \
    --dilation \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
