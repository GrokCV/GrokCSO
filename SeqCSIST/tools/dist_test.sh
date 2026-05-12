#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29999}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}

## The following comment is the terminal command for running MMDetection 3.x version on a single node with multiple GPUs.

## 终端运行指令
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
#     --nproc_per_node=4 \
#     --master_port=29999 \
#     tools/test.py \
#     configs/configs/DeRefNet.py \
#     work_dir/DeRefNet/best_cso_metric_mAP_epoch_20.pth \
#     --launcher pytorch


## 旧版本

# CONFIG=$1
# CHECKPOINT=$2
# GPUS=$3
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29800}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     $(dirname "$0")/test.py \
#     $CONFIG \
#     $CHECKPOINT \
#     --launcher pytorch \
#     ${@:4}

# ## CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/configs/istanet.py work_dirs/istanet/best_cso_metric_mAP_epoch_190.pth 4
